package org.apache.lucene.codecs.sandbox;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
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
  static final String CENTROIDS_META_EXTENSION = "vemcbq";
  static final String CENTROIDS_DATA_EXTENSION = "veccbq";
  static final int VERSION_CURRENT = 0;

  /**
   * Build parameters for a span index.
   *
   * @param centroidFraction fraction of input points to choose as centroids.
   * @param centroidSearchCandidates how many candidate centroids to choose for each point.
   * @param centroidEpsilon controls centroid expansion, used to adjust the maximum distance to
   *     allow for any secondary centroids based on primary centroid distance.
   * @param maxCentroids select no more than this many centroids for each point.
   */
  public record BuildParams(
      float centroidFraction,
      int centroidSearchCandidates,
      float centroidEpsilon,
      int maxCentroids) {
    BuildParams() {
      this(0.16f, 10, 10.0f, 8);
    }
  }

  private final Lucene99FlatVectorsFormat rawFlatVectorsFormat = new Lucene99FlatVectorsFormat();
  private final BinaryQuantizedFlatVectorsFormat bqFlatVectorsFormat =
      new BinaryQuantizedFlatVectorsFormat();
  private final int maxConn;
  private final int beamWidth;
  private final BuildParams buildParams;
  private final int numMergeWorkers;
  private final TaskExecutor mergeExec;

  public SpannBinaryQuantizedVectorsFormat() {
    this(
        HnswBinaryQuantizedVectorsFormat.DEFAULT_MAX_CONN,
        HnswBinaryQuantizedVectorsFormat.DEFAULT_BEAM_WIDTH,
        new BuildParams(),
        HnswBinaryQuantizedVectorsFormat.DEFAULT_NUM_MERGE_WORKER,
        null);
  }

  public SpannBinaryQuantizedVectorsFormat(
      int M, int beamWidth, BuildParams params, int numMergeWorkers, ExecutorService mergeExec) {
    super("SpannBinaryQuantizedVectorsFormat");
    this.maxConn = M;
    this.beamWidth = beamWidth;
    this.buildParams = params;
    this.numMergeWorkers = numMergeWorkers;
    this.mergeExec = mergeExec != null ? new TaskExecutor(mergeExec) : null;
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
            new BinaryQuantizedFlatVectorsWriter(
                state, CENTROIDS_META_EXTENSION, CENTROIDS_DATA_EXTENSION),
            null,
            this.numMergeWorkers,
            this.mergeExec),
        this.buildParams);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new SpannBinaryQuantizedVectorsReader(
        state,
        this.rawFlatVectorsFormat.fieldsReader(state),
        this.bqFlatVectorsFormat.fieldsReader(state),
        new HnswBinaryQuantizedVectorsReader(
            state,
            new BinaryQuantizedFlatVectorsReader(
                state, CENTROIDS_META_EXTENSION, CENTROIDS_DATA_EXTENSION),
            null));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 4096;
  }
}
