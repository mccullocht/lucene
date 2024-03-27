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
import org.apache.lucene.codecs.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Format supporting vector quantization, storage, and retrieval
 *
 * @lucene.experimental
 */
public final class BinaryQuantizedFlatVectorsFormat extends FlatVectorsFormat {
  static final String NAME = "BinaryQuantizedFlatVectorsFormat";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;
  static final String META_CODEC_NAME = "BinaryQuantizedFlatVectorsFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "BinaryQuantizedFlatVectorsFormatData";
  static final String META_EXTENSION = "vemfbq";
  static final String VECTOR_DATA_EXTENSION = "vecbq";

  static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

  static final FlatVectorsFormat rawVectorFormat = new Lucene99FlatVectorsFormat();

  /** Constructs a format using default graph construction parameters */
  public BinaryQuantizedFlatVectorsFormat() {}

  @Override
  public String toString() {
    return NAME + "(name=" + NAME + ", rawVectorFormat=" + rawVectorFormat + ")";
  }

  @Override
  public BinaryQuantizedFlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new BinaryQuantizedFlatVectorsWriter(state, rawVectorFormat.fieldsWriter(state));
  }

  @Override
  public BinaryQuantizedFlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new BinaryQuantizedFlatVectorsReader(state, rawVectorFormat.fieldsReader(state));
  }

  public FlatVectorsFormat getRawVectorFormat() {
    return rawVectorFormat;
  }
}
