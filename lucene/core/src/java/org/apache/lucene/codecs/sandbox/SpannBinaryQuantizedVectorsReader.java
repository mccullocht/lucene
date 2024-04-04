package org.apache.lucene.codecs.sandbox;

import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import java.io.IOException;
import java.util.HashSet;
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
    var seenOrds = new HashSet<Integer>();
    int collected =
        scoreCentroid(
            indexAccess,
            basePlOffset,
            centroidDocs[0].doc,
            scorer,
            acceptOrds,
            seenOrds,
            collector);
    float maxDistance = (1.0f / centroidDocs[0].score) * (1.0f + queryEpsilon);
    for (int i = 1; i < centroidDocs.length; i++) {
      ScoreDoc secondaryCentroid = centroidDocs[i];
      // Prune out secondary centroids if we've already collected k hits and the score exceeds the
      // maximum distance.
      //if (collected >= knnCollector.k() && (1.0f / secondaryCentroid.score) > maxDistance) {
      if ((1.0f / secondaryCentroid.score) > maxDistance) {
        if (collected >= knnCollector.k()) {
          System.err.println("break at " + i);
          break;
        }
        System.err.println("collected: " + collected + " k: " + knnCollector.k() + " primaryDistance: " + (1.0f / centroidDocs[0].score) + " maxDistance: " + maxDistance + " i: " + i + " distance: " + (1.0f / secondaryCentroid.score));
      }
      scoreCentroid(
          indexAccess,
          basePlOffset,
          secondaryCentroid.doc,
          scorer,
          acceptOrds,
          seenOrds,
          collector);
    }
  }

  private static int scoreCentroid(
      RandomAccessInput indexAccess,
      long basePlOffset,
      int centroid,
      RandomVectorScorer scorer,
      Bits acceptOrds,
      HashSet<Integer> seenOrds,
      KnnCollector collector)
      throws IOException {
    int hitsStart = indexAccess.readInt(centroid * Integer.BYTES);
    int hitsEnd = indexAccess.readInt((centroid + 1) * Integer.BYTES);
    int collected = 0;
    for (int j = hitsStart; j < hitsEnd; j++) {
      int hitOrd = indexAccess.readInt(basePlOffset + j * Integer.BYTES);
      if (seenOrds.add(hitOrd) && (acceptOrds == null || acceptOrds.get(hitOrd))) {
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
