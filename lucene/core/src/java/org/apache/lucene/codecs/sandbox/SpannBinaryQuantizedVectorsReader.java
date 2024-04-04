package org.apache.lucene.codecs.sandbox;

import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import java.io.IOException;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SparseFixedBitSet;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/** Placate tidy */
public class SpannBinaryQuantizedVectorsReader extends KnnVectorsReader
    implements BinaryQuantizedVectorsReader {
  private static final long SHALLOW_SIZE =
      shallowSizeOfInstance(SpannBinaryQuantizedVectorsReader.class);

  private final FlatVectorsReader rawFlatVectorsReader;
  private final BinaryQuantizedFlatVectorsReader bqFlatVectorsReader;
  private final HnswBinaryQuantizedVectorsReader centroidsHnswReader;
  private final IndexInput index;

  public SpannBinaryQuantizedVectorsReader(
      SegmentReadState state,
      FlatVectorsReader rawFlatVectorsReader,
      BinaryQuantizedFlatVectorsReader bqFlatVectorsReader,
      HnswBinaryQuantizedVectorsReader centroidsHnswReader)
      throws IOException {
    this.rawFlatVectorsReader = rawFlatVectorsReader;
    this.bqFlatVectorsReader = bqFlatVectorsReader;
    this.centroidsHnswReader = centroidsHnswReader;

    String indexFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            SpannBinaryQuantizedVectorsFormat.INDEX_EXTENSION);
    boolean success = false;
    try {
      IndexInput in = state.directory.openInput(indexFileName, state.context);
      CodecUtil.checkIndexHeader(
          in,
          SpannBinaryQuantizedVectorsFormat.INDEX_CODEC_NAME,
          0,
          0,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      long offset = in.getFilePointer() + (Integer.BYTES - (in.getFilePointer() % Integer.BYTES));
      long length = in.length() - offset;
      // XXX i'm pretty sure this leaks memory with MemorySegmentIndexInput. Hella annoying.
      this.index = in.slice("spann-index", offset, length);
      CodecUtil.retrieveChecksum(in);
      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return this.rawFlatVectorsReader.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return this.rawFlatVectorsReader.getByteVectorValues(field);
  }

  @Override
  public BinaryVectorValues getBinaryVectorValues(String fieldName) throws IOException {
    return this.bqFlatVectorsReader.getBinaryVectorValues(fieldName);
  }

  private static final float queryEpsilon;
  private static final int maxCentroids;

  static {
    String rawQueryEpsilon = System.getenv("SPANN_QUERY_EPSILON");
    if (rawQueryEpsilon != null && !rawQueryEpsilon.equals("null")) {
      try {
        queryEpsilon = Float.valueOf(rawQueryEpsilon);
      } catch (NumberFormatException e) {
        throw new IllegalArgumentException(e);
      }
    } else {
      queryEpsilon = 0.0f;
    }

    String rawMaxCentroids = System.getenv("SPANN_MAX_CENTROIDS");
    if (rawMaxCentroids != null && !rawMaxCentroids.equals("null")) {
      try {
        maxCentroids = Integer.valueOf(rawMaxCentroids);
      } catch (NumberFormatException e) {
        throw new IllegalArgumentException(e);
      }
    } else {
      maxCentroids = Integer.MAX_VALUE;
    }
  }

  private static float scoreToDistance(double score) {
    // The score is 1.0 / (1 + distance); invert this to extract distance.
    return (float) ((1.0 / score) - 1);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    // XXX hack to avoid ordinal translation for centroids.
    ScoreDoc[] centroidDocs =
        this.centroidsHnswReader.searchCentroids(field, target, knnCollector.k(), null).scoreDocs;
    if (centroidDocs.length == 0) {
      return;
    }
    var indexAccess = this.index.randomAccessSlice(0, this.index.length());
    // NB: we quantize the vector a second time here, which is fun.
    var scorer = this.bqFlatVectorsReader.getRandomVectorScorer(field, target);
    final KnnCollector collector =
        new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);
    Bits acceptOrds = scorer.getAcceptOrds(acceptDocs);
    int numCentroids = this.centroidsHnswReader.getBinaryVectorValues(field).size();
    int basePlOffset = (numCentroids + 1) * Integer.BYTES;
    var seenOrds = new SparseFixedBitSet(scorer.maxOrd());
    assert centroidDocs[0].doc < numCentroids;
    int collected =
        scoreCentroid(
            indexAccess, basePlOffset, centroidDocs[0], scorer, acceptOrds, seenOrds, collector);
    float maxDistance = scoreToDistance(centroidDocs[0].score) * (1.0f + queryEpsilon);
    for (int i = 1; i < centroidDocs.length; i++) {
      ScoreDoc secondaryCentroid = centroidDocs[i];
      // Prune out secondary centroids if we've already collected k hits and the score exceeds the
      // maximum distance.
      if (collected >= knnCollector.k()
          && ((1.0f / secondaryCentroid.score) > maxDistance || i > maxCentroids)) {
        break;
      }
      collected +=
          scoreCentroid(
              indexAccess,
              basePlOffset,
              secondaryCentroid,
              scorer,
              acceptOrds,
              seenOrds,
              collector);
    }
  }

  private static int scoreCentroid(
      RandomAccessInput indexAccess,
      long basePlOffset,
      ScoreDoc centroid,
      RandomVectorScorer scorer,
      Bits acceptOrds,
      SparseFixedBitSet seenOrds,
      KnnCollector collector)
      throws IOException {
    int hitsStart = indexAccess.readInt(centroid.doc * Integer.BYTES * 2);
    int hitsEnd = indexAccess.readInt((centroid.doc + 1) * Integer.BYTES * 2);
    float centroidDistance = scoreToDistance(centroid.score);
    long offset = basePlOffset + (hitsStart * Integer.BYTES * 2);
    int collected = 0;
    for (int j = hitsStart; j < hitsEnd; j++, offset += Integer.BYTES * 2) {
      int hitOrd = indexAccess.readInt(offset);
      float hitDistance = Float.intBitsToFloat(indexAccess.readInt(offset + Integer.BYTES));
      // Use reverse triangle inequality to compute a lower bound for distance, then use that value
      // to compute an upper bound for score. If the score is not competitive then don't score.
      float upperBoundScore = 1.0f / (1.0f + Math.abs(centroidDistance - hitDistance));
      if (!seenOrds.getAndSet(hitOrd)
          && (acceptOrds == null || acceptOrds.get(hitOrd))
          && upperBoundScore > collector.minCompetitiveSimilarity()) {
        collector.collect(hitOrd, scorer.score(hitOrd));
        collected += 1;
      }
    }
    return collected;
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    throw new UnsupportedOperationException("unimplemented");
  }

  @Override
  public void checkIntegrity() throws IOException {
    this.rawFlatVectorsReader.checkIntegrity();
    this.bqFlatVectorsReader.checkIntegrity();
    this.centroidsHnswReader.checkIntegrity();
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(
        this.rawFlatVectorsReader, this.bqFlatVectorsReader, this.centroidsHnswReader, this.index);
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE
        + this.rawFlatVectorsReader.ramBytesUsed()
        + this.bqFlatVectorsReader.ramBytesUsed()
        + this.centroidsHnswReader.ramBytesUsed();
  }
}
