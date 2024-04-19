package org.apache.lucene.codecs.sandbox;

import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** Placate tidy */
public class SpannBinaryQuantizedVectorsWriter extends KnnVectorsWriter {
  private static long SHALLOW_SIZE = shallowSizeOfInstance(SpannBinaryQuantizedVectorsWriter.class);

  private final SegmentWriteState state;
  private final FlatVectorsWriter rawFlatVectorsWriter;
  private final BinaryQuantizedFlatVectorsWriter bqFlatVectorsWriter;
  private final HnswBinaryQuantizedVectorsWriter bqHnswVectorsWriter;
  private final SpannBinaryQuantizedVectorsFormat.BuildParams params;

  private final List<FieldWriter<?>> fields = new ArrayList<>();
  private final IndexOutput index;
  private boolean finished = false;

  public SpannBinaryQuantizedVectorsWriter(
      SegmentWriteState state,
      FlatVectorsWriter rawFlatVectorsWriter,
      BinaryQuantizedFlatVectorsWriter bqFlatVectorsWriter,
      HnswBinaryQuantizedVectorsWriter bqHnswVectorsWriter,
      SpannBinaryQuantizedVectorsFormat.BuildParams params)
      throws IOException {
    this.state = state;
    this.rawFlatVectorsWriter = rawFlatVectorsWriter;
    this.bqFlatVectorsWriter = bqFlatVectorsWriter;
    this.bqHnswVectorsWriter = bqHnswVectorsWriter;
    this.params = params;

    String indexFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            SpannBinaryQuantizedVectorsFormat.INDEX_EXTENSION);
    boolean success = false;
    try {
      this.index = this.state.directory.createOutput(indexFileName, state.context);
      CodecUtil.writeIndexHeader(
          this.index,
          SpannBinaryQuantizedVectorsFormat.INDEX_CODEC_NAME,
          SpannBinaryQuantizedVectorsFormat.VERSION_CURRENT,
          this.state.segmentInfo.getId(),
          this.state.segmentSuffix);
      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var writer =
        FieldWriter.create(
            fieldInfo,
            this.rawFlatVectorsWriter.addField(fieldInfo, null),
            this.bqFlatVectorsWriter.addField(fieldInfo, null),
            this.bqHnswVectorsWriter.addField(fieldInfo));
    this.fields.add(writer);
    return writer;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    if (this.fields.size() != 1) {
      throw new UnsupportedOperationException("exactly one field");
    }
    if (sortMap != null) {
      throw new UnsupportedOperationException("sorting is unimplemented");
    }

    for (var field : this.fields) {
      writeField(field);
    }
    // Flush everything after writing the index in case some of the writers *cough hnsw* destroy
    // some of their state.
    this.rawFlatVectorsWriter.flush(maxDoc, sortMap);
    this.bqFlatVectorsWriter.flush(maxDoc, sortMap);
    this.bqHnswVectorsWriter.flush(maxDoc, sortMap);
  }

  // XXX this will be a disaster when merging because small segments will skew the centroids with
  // random garbage. Fine for this though!
  private List<long[]> selectCentroids(List<long[]> allPoints) {
    record VectorBitCount(long[] vector, int count) {
      VectorBitCount(long[] vector) {
        this(vector, (int) Arrays.stream(vector).map(v -> Long.bitCount(v)).sum());
      }
    }
    List<VectorBitCount> vectorBitCounts =
        allPoints.stream()
            .map(v -> new VectorBitCount(v))
            .sorted(Comparator.comparingInt(vbc -> vbc.count))
            .collect(Collectors.toList());
    int numCentroids = (int) (allPoints.size() * this.params.centroidFraction());
    return vectorBitCounts
        .subList(
            (allPoints.size() / 2) - (numCentroids / 2),
            (allPoints.size() / 2) + (numCentroids / 2))
        .stream()
        .map(vbc -> vbc.vector)
        .collect(Collectors.toList());
  }

  private void writeField(FieldWriter<?> field) throws IOException {
    List<long[]> allPoints = field.bqFlatWriter.getVectors();

    for (var vector : selectCentroids(allPoints)) {
      field.bqHnswWriter.addValue(vector);
    }

    RandomAccessVectorValues<long[]> centroidValues = field.bqHnswWriter.newRandomAccessValues();
    // NB: in general if this happens the segment is smol and scanning is fine.
    if (centroidValues.size() == 0) {
      throw new IllegalStateException("empty centroids");
    }
    ArrayList<ArrayList<Integer>> centroidPls = new ArrayList<>(centroidValues.size());
    for (int i = 0; i < centroidValues.size(); i++) {
      centroidPls.add(new ArrayList<>());
    }
    OnHeapHnswGraph centroidGraph = field.bqHnswWriter.getGraph();
    for (int i = 0; i < allPoints.size(); i++) {
      var collector =
          HnswGraphSearcher.search(
              new BinaryQuantizedRandomVectorScorer(centroidValues, allPoints.get(i)),
              this.params.centroidSearchCandidates(),
              centroidGraph,
              null,
              Integer.MAX_VALUE);
      ScoreDoc[] centroidCandidates = collector.topDocs().scoreDocs;
      ScoreDoc primaryCentroid = centroidCandidates[0];
      centroidPls.get(primaryCentroid.doc).add(i);
      // The score is the inversion of the distance metric, un-invert to get back to a distance.
      float minScore = primaryCentroid.score / this.params.centroidEpsilon();

      int numCentroids = 1;
      for (int j = 1; j < centroidCandidates.length; j++) {
        ScoreDoc secondaryCentroid = centroidCandidates[j];
        if (secondaryCentroid.score < minScore) {
          break;
        }
        var scorer =
            new BinaryQuantizedRandomVectorScorer(
                centroidValues, centroidValues.vectorValue(secondaryCentroid.doc));
        if (scorer.score(centroidCandidates[j - 1].doc) < secondaryCentroid.score) {
          centroidPls.get(secondaryCentroid.doc).add(i);
          numCentroids += 1;
          if (numCentroids >= this.params.maxCentroids()) {
            break;
          }
        }
      }
    }

    // XXX to be complete we need to write metadata support multifield even though functionally
    // everything is wrapped and each KnnVectorWriter only has one field.
    this.index.alignFilePointer(Integer.BYTES);
    int totalHits = 0;
    for (int i = 0; i < centroidPls.size(); i++) {
      // NB: every centroid should have _something_ because the centroid appears in the data set so
      // there should be at least one exact match hit.
      this.index.writeInt(totalHits);
      totalHits += centroidPls.get(i).size();
    }
    this.index.writeInt(totalHits);

    for (ArrayList<Integer> pl : centroidPls) {
      for (int hit : pl) {
        this.index.writeInt(hit);
      }
    }
  }

  @Override
  public void finish() throws IOException {
    if (this.finished) {
      throw new IllegalStateException("already finished");
    }

    this.finished = true;
    this.rawFlatVectorsWriter.finish();
    this.bqFlatVectorsWriter.finish();
    this.bqHnswVectorsWriter.finish();

    if (this.index != null) {
      CodecUtil.writeFooter(this.index);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    throw new UnsupportedOperationException("unimplemented");
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(
        this.index, this.rawFlatVectorsWriter, this.bqFlatVectorsWriter, this.bqHnswVectorsWriter);
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE + this.fields.stream().map(f -> f.ramBytesUsed()).reduce(0L, Long::sum);
  }

  private abstract static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
    private static final long SHALLOW_SIZE = shallowSizeOfInstance(FieldWriter.class);

    private final KnnFieldVectorsWriter<T> rawFlatWriter;
    private final BinaryQuantizedFlatVectorsWriter.FieldWriter<T> bqFlatWriter;
    private final HnswBinaryQuantizedVectorsWriter.FieldWriter<T> bqHnswWriter;

    @SuppressWarnings("unchecked")
    static FieldWriter<?> create(
        FieldInfo fieldInfo,
        KnnFieldVectorsWriter<?> rawFlatWriter,
        BinaryQuantizedFlatVectorsWriter.FieldWriter<?> bqFlatWriter,
        HnswBinaryQuantizedVectorsWriter.FieldWriter<?> bqHnswWriter) {
      return switch (fieldInfo.getVectorEncoding()) {
        case BYTE -> new FieldWriter<>(
            (KnnFieldVectorsWriter<byte[]>) rawFlatWriter,
            (BinaryQuantizedFlatVectorsWriter.FieldWriter<byte[]>) bqFlatWriter,
            (HnswBinaryQuantizedVectorsWriter.FieldWriter<byte[]>) bqHnswWriter) {
          @Override
          protected int vectorHash(byte[] vectorValue) {
            return Arrays.hashCode(vectorValue);
          }
        };
        case FLOAT32 -> new FieldWriter<>(
            (KnnFieldVectorsWriter<float[]>) rawFlatWriter,
            (BinaryQuantizedFlatVectorsWriter.FieldWriter<float[]>) bqFlatWriter,
            (HnswBinaryQuantizedVectorsWriter.FieldWriter<float[]>) bqHnswWriter) {
          @Override
          protected int vectorHash(float[] vectorValue) {
            return Arrays.hashCode(vectorValue);
          }
        };
      };
    }

    protected FieldWriter(
        KnnFieldVectorsWriter<T> rawFlatWriter,
        BinaryQuantizedFlatVectorsWriter.FieldWriter<T> bqFlatWriter,
        HnswBinaryQuantizedVectorsWriter.FieldWriter<T> bqHnswWriter) {
      this.rawFlatWriter = rawFlatWriter;
      this.bqFlatWriter = bqFlatWriter;
      this.bqHnswWriter = bqHnswWriter;
    }

    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
      this.rawFlatWriter.addValue(docID, vectorValue);
      this.bqFlatWriter.addValue(docID, vectorValue);
      // NB: centroids chosen at the end.
    }

    protected abstract int vectorHash(T vectorValue);

    @Override
    public T copyValue(T vectorValue) {
      throw new UnsupportedOperationException("unimplemented");
    }

    @Override
    public long ramBytesUsed() {
      return SHALLOW_SIZE
          + rawFlatWriter.ramBytesUsed()
          + bqFlatWriter.ramBytesUsed()
          + bqHnswWriter.ramBytesUsed();
    }
  }
}
