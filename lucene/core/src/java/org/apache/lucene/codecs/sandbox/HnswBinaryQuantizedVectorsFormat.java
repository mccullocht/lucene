/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.sandbox;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import org.apache.lucene.codecs.FlatVectorsFormat;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HnswGraph;

/**
 * Vector format that uses an HNSW graph for vector search but over binary quantized vectors. This
 * yields candidate with quantized scores, you will almost certainly want to re-score before
 * returning them to users.
 *
 * @lucene.experimental
 */
public final class HnswBinaryQuantizedVectorsFormat extends KnnVectorsFormat {
  /**
   * A maximum configurable maximum max conn.
   *
   * <p>NOTE: We eagerly populate `float[MAX_CONN*2]` and `int[MAX_CONN*2]`, so exceptionally large
   * numbers here will use an inordinate amount of heap
   */
  static final int MAXIMUM_MAX_CONN = 512;

  /** Default number of maximum connections per node */
  public static final int DEFAULT_MAX_CONN = 16;

  /**
   * The maximum size of the queue to maintain while searching during graph construction This
   * maximum value preserves the ratio of the DEFAULT_BEAM_WIDTH/DEFAULT_MAX_CONN i.e. `6.25 * 16 =
   * 3200`
   */
  static final int MAXIMUM_BEAM_WIDTH = 3200;

  /**
   * Default number of the size of the queue maintained while searching during a graph construction.
   */
  public static final int DEFAULT_BEAM_WIDTH = 100;

  /** Default to use single thread merge */
  public static final int DEFAULT_NUM_MERGE_WORKER = 1;

  /**
   * Controls how many of the nearest neighbor candidates are connected to the new node. Defaults to
   * {@link Lucene99HnswVectorsFormat#DEFAULT_MAX_CONN}. See {@link HnswGraph} for more details.
   */
  private final int maxConn;

  /**
   * The number of candidate neighbors to track while searching the graph for each newly inserted
   * node. Defaults to {@link Lucene99HnswVectorsFormat#DEFAULT_BEAM_WIDTH}. See {@link HnswGraph}
   * for details.
   */
  private final int beamWidth;

  /** The format for storing, reading, merging vectors on disk */
  private final FlatVectorsFormat flatVectorsFormat = new BinaryQuantizedFlatVectorsFormat();

  private final int numMergeWorkers;
  private final TaskExecutor mergeExec;

  /** Constructs a format using default graph construction parameters */
  public HnswBinaryQuantizedVectorsFormat() {
    this(DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
  }

  /**
   * Constructs a format using the given graph construction parameters.
   *
   * @param maxConn the maximum number of connections to a node in the HNSW graph
   * @param beamWidth the size of the queue maintained during graph construction.
   */
  public HnswBinaryQuantizedVectorsFormat(int maxConn, int beamWidth) {
    this(maxConn, beamWidth, DEFAULT_NUM_MERGE_WORKER, null);
  }

  /**
   * Constructs a format using the given graph construction parameters and scalar quantization.
   *
   * @param maxConn the maximum number of connections to a node in the HNSW graph
   * @param beamWidth the size of the queue maintained during graph construction.
   * @param numMergeWorkers number of workers (threads) that will be used when doing merge. If
   *     larger than 1, a non-null {@link ExecutorService} must be passed as mergeExec
   * @param mergeExec the {@link ExecutorService} that will be used by ALL vector writers that are
   *     generated by this format to do the merge
   */
  public HnswBinaryQuantizedVectorsFormat(
      int maxConn, int beamWidth, int numMergeWorkers, ExecutorService mergeExec) {
    super("HnswBinaryQuantizedVectorsFormat");
    if (maxConn <= 0 || maxConn > MAXIMUM_MAX_CONN) {
      throw new IllegalArgumentException(
          "maxConn must be positive and less than or equal to "
              + MAXIMUM_MAX_CONN
              + "; maxConn="
              + maxConn);
    }
    if (beamWidth <= 0 || beamWidth > MAXIMUM_BEAM_WIDTH) {
      throw new IllegalArgumentException(
          "beamWidth must be positive and less than or equal to "
              + MAXIMUM_BEAM_WIDTH
              + "; beamWidth="
              + beamWidth);
    }
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    if (numMergeWorkers > 1 && mergeExec == null) {
      throw new IllegalArgumentException(
          "No executor service passed in when " + numMergeWorkers + " merge workers are requested");
    }
    if (numMergeWorkers == 1 && mergeExec != null) {
      throw new IllegalArgumentException(
          "No executor service is needed as we'll use single thread to merge");
    }
    this.numMergeWorkers = numMergeWorkers;
    if (mergeExec != null) {
      this.mergeExec = new TaskExecutor(mergeExec);
    } else {
      this.mergeExec = null;
    }
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Lucene99HnswVectorsWriter(
        state,
        maxConn,
        beamWidth,
        flatVectorsFormat.fieldsWriter(state),
        numMergeWorkers,
        mergeExec);
  }

  private static final boolean RESCORE = Boolean.parseBoolean(System.getenv("BQ_SEGMENT_RESCORE"));
  private static final float OVERSAMPLE;

  static {
    float oversample;
    try {
      oversample = Float.parseFloat(System.getenv("BQ_SEGMENT_RESCORE_OVERSAMPLE"));
    } catch (NullPointerException | NumberFormatException e) {
      oversample = 1.0f;
    }
    OVERSAMPLE = oversample;
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    var flatVectorsReader = this.flatVectorsFormat.fieldsReader(state);
    return RESCORE
        ? new Reader(state, flatVectorsReader)
        : new Lucene99HnswVectorsReader(state, flatVectorsReader);
  }

  private static class Reader extends KnnVectorsReader {
    private final Lucene99HnswVectorsReader inner;
    private final FlatVectorsReader flatVectorsReader;
    private final Map<String, VectorSimilarityFunction> fields;

    Reader(SegmentReadState state, FlatVectorsReader flatVectorsReader) throws IOException {
      this.inner = new Lucene99HnswVectorsReader(state, flatVectorsReader);
      this.flatVectorsReader = flatVectorsReader;
      this.fields = new HashMap<>();
      for (FieldInfo fi : state.fieldInfos) {
        // NB: hasVectors() is a really unfortunate name.
        if (fi.hasVectorValues()) {
          this.fields.put(fi.name, fi.getVectorSimilarityFunction());
        }
      }
    }

    @Override
    public void checkIntegrity() throws IOException {
      this.inner.checkIntegrity();
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
      return this.inner.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
      return this.inner.getByteVectorValues(field);
    }

    // XXX it would be better to do this above the segment level:
    // * as-is this will score up to num_segments times as many documents as we'd like.
    // * this happens after ord -> doc translation so it would be busted if there were multiple
    //   vectors attached to a doc.
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      var bqCollector = bqCollector(knnCollector);
      this.inner.search(field, target, bqCollector, acceptDocs);
      var vectorValues = this.flatVectorsReader.getFloatVectorValues(field);
      var sim = this.fields.get(field);
      // Sort in doc order to join with values.
      Arrays.sort(bqCollector.topDocs().scoreDocs, Comparator.comparingInt(sd -> sd.doc));
      // XXX this is going to generate wrong data related to early termination.
      for (var scoreDoc : bqCollector.topDocs().scoreDocs) {
        vectorValues.advance(scoreDoc.doc);
        if (vectorValues.docID() == scoreDoc.doc) {
          knnCollector.collect(scoreDoc.doc, sim.compare(target, vectorValues.vectorValue()));
        }
      }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      var bqCollector = bqCollector(knnCollector);
      this.inner.search(field, target, bqCollector, acceptDocs);
      var vectorValues = this.flatVectorsReader.getByteVectorValues(field);
      var sim = this.fields.get(field);
      // Sort in doc order to join with values.
      Arrays.sort(bqCollector.topDocs().scoreDocs, Comparator.comparingInt(sd -> sd.doc));
      // XXX this is going to generate wrong data related to early termination.
      for (var scoreDoc : bqCollector.topDocs().scoreDocs) {
        vectorValues.advance(scoreDoc.doc);
        if (vectorValues.docID() == scoreDoc.doc) {
          knnCollector.collect(scoreDoc.doc, sim.compare(target, vectorValues.vectorValue()));
        }
      }
    }

    @Override
    public void close() throws IOException {
      this.inner.close();
    }

    @Override
    public long ramBytesUsed() {
      return this.inner.ramBytesUsed();
    }

    private static KnnCollector bqCollector(KnnCollector collector) {
      return new TopKnnCollector((int) (collector.k() * OVERSAMPLE), (int) collector.visitLimit());
    }
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return "HnswBinaryQuantizedVectorsFormat(name=HnswBinaryQuantizedVectorsFormat, maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", flatVectorFormat="
        + flatVectorsFormat
        + ")";
  }
}
