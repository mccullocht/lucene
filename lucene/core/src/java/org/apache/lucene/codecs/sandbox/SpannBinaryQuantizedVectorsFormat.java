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

  static final int DEFAULT_CENTROID_CANDIDATES = 10;
  static final int DEFAULT_MAX_CENTROIDS = 8;
  static final float DEFAULT_CENTROID_EPSILON = 0.0f;

  private final Lucene99FlatVectorsFormat rawFlatVectorsFormat = new Lucene99FlatVectorsFormat();
  private final BinaryQuantizedFlatVectorsFormat bqFlatVectorsFormat =
      new BinaryQuantizedFlatVectorsFormat();
  private final int maxConn;
  private final int beamWidth;
  private final int centroidCandidates;
  private final int maxCentroids;
  private final float centroidEpsilon;
  private final int numMergeWorkers;
  private final TaskExecutor mergeExec;

  public SpannBinaryQuantizedVectorsFormat() {
    this(
        HnswBinaryQuantizedVectorsFormat.DEFAULT_MAX_CONN,
        HnswBinaryQuantizedVectorsFormat.DEFAULT_BEAM_WIDTH,
        DEFAULT_CENTROID_CANDIDATES,
        DEFAULT_MAX_CENTROIDS,
        DEFAULT_CENTROID_EPSILON,
        HnswBinaryQuantizedVectorsFormat.DEFAULT_NUM_MERGE_WORKER,
        null);
  }

  public SpannBinaryQuantizedVectorsFormat(
      int M,
      int beamWidth,
      int centroidCandidates,
      int maxCentroids,
      float centroidEpsilon,
      int numMergeWorkers,
      ExecutorService mergeExec) {
    super("SpannBinaryQuantizedVectorsFormat");
    this.maxConn = M;
    this.beamWidth = beamWidth;
    this.centroidCandidates = centroidCandidates;
    this.maxCentroids = maxCentroids;
    this.centroidEpsilon = centroidEpsilon;
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
        this.centroidCandidates,
        this.maxCentroids,
        this.centroidEpsilon);
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
