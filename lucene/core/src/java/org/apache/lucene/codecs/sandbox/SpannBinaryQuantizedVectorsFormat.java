package org.apache.lucene.codecs.sandbox;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;

/** Placate tidy */
public class SpannBinaryQuantizedVectorsFormat extends KnnVectorsFormat {
  static final String INDEX_EXTENSION = "vespidx";
  static final String INDEX_CODEC_NAME = "SpannBinaryQuantizedVectorsFormatIndex";
  static final int VERSION_CURRENT = 0;

  private final Lucene99FlatVectorsFormat rawFlatVectorsFormat = new Lucene99FlatVectorsFormat();
  private final BinaryQuantizedFlatVectorsFormat bqFlatVectorsFormat =
      new BinaryQuantizedFlatVectorsFormat();
  private final int maxConn;
  private final int beamWidth;
  private final int numMergeWorkers;
  private final TaskExecutor mergeExec;

  public SpannBinaryQuantizedVectorsFormat() {
    this(
        HnswBinaryQuantizedVectorsFormat.DEFAULT_MAX_CONN,
        HnswBinaryQuantizedVectorsFormat.DEFAULT_BEAM_WIDTH,
        HnswBinaryQuantizedVectorsFormat.DEFAULT_NUM_MERGE_WORKER,
        null);
  }

  public SpannBinaryQuantizedVectorsFormat(
      int M, int beamWidth, int numMergeWorkers, TaskExecutor mergeExec) {
    super("SpannBinaryQuantizedVectorsFormat");
    this.maxConn = M;
    this.beamWidth = beamWidth;
    this.numMergeWorkers = numMergeWorkers;
    this.mergeExec = mergeExec;
  }

  @Override
  public SpannBinaryQuantizedVectorsWriter fieldsWriter(SegmentWriteState state)
      throws IOException {
    return new SpannBinaryQuantizedVectorsWriter(
        state,
        this.rawFlatVectorsFormat.fieldsWriter(state),
        this.bqFlatVectorsFormat.fieldsWriter(state),
        new HnswBinaryQuantizedVectorsWriter(
            state,
            this.maxConn,
            this.beamWidth,
            new BinaryQuantizedFlatVectorsWriter(state, "vemcbq", "veccbq"),
            null,
            this.numMergeWorkers,
            this.mergeExec));
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    throw new UnsupportedOperationException("unimplemented");
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 4096;
  }
}
