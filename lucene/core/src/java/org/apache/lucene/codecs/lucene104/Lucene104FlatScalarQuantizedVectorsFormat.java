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
package org.apache.lucene.codecs.lucene104;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatKnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatKnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

/**
 * A {@link KnnVectorsFormat} that stores vectors using {@link Lucene104ScalarQuantizedVectorsFormat}
 * and performs exhaustive (flat) nearest-neighbor search at query time. Unlike {@link
 * Lucene104HnswScalarQuantizedVectorsFormat}, no HNSW graph is built, so indexing is cheaper but
 * search cost scales linearly with the number of indexed vectors.
 *
 * @lucene.experimental
 */
public final class Lucene104FlatScalarQuantizedVectorsFormat extends KnnVectorsFormat {

  public static final String NAME = "Lucene104FlatScalarQuantizedVectorsFormat";

  private final Lucene104ScalarQuantizedVectorsFormat flatFormat;

  /** Creates a new instance with the default {@link ScalarEncoding#UNSIGNED_BYTE} encoding. */
  public Lucene104FlatScalarQuantizedVectorsFormat() {
    this(new Lucene104ScalarQuantizedVectorsFormat());
  }

  /** Creates a new instance with the given scalar quantization encoding. */
  public Lucene104FlatScalarQuantizedVectorsFormat(ScalarEncoding encoding) {
    this(new Lucene104ScalarQuantizedVectorsFormat(encoding));
  }

  private Lucene104FlatScalarQuantizedVectorsFormat(
      Lucene104ScalarQuantizedVectorsFormat flatFormat) {
    super(NAME);
    this.flatFormat = flatFormat;
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new FlatKnnVectorsWriter(flatFormat.fieldsWriter(state));
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new FlatKnnVectorsReader(flatFormat.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return flatFormat.getMaxDimensions(fieldName);
  }

  @Override
  public String toString() {
    return "Lucene104FlatScalarQuantizedVectorsFormat(flatFormat=" + flatFormat + ")";
  }
}
