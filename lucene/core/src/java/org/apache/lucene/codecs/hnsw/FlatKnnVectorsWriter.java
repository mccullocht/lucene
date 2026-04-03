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
package org.apache.lucene.codecs.hnsw;

import java.io.IOException;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOUtils;

/**
 * A {@link KnnVectorsWriter} that delegates all operations to an underlying {@link
 * FlatVectorsWriter}. This is useful for building a {@link
 * org.apache.lucene.codecs.KnnVectorsFormat} backed purely by flat (exhaustive) vector storage
 * without an additional graph index.
 *
 * @lucene.experimental
 */
public final class FlatKnnVectorsWriter extends KnnVectorsWriter {

  private final FlatVectorsWriter flatWriter;

  /** Creates a new writer delegating to the given flat writer. */
  public FlatKnnVectorsWriter(FlatVectorsWriter flatWriter) {
    this.flatWriter = flatWriter;
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    return flatWriter.addField(fieldInfo);
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    flatWriter.flush(maxDoc, sortMap);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatWriter.mergeOneField(fieldInfo, mergeState);
  }

  @Override
  public void finish() throws IOException {
    flatWriter.finish();
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(flatWriter);
  }

  @Override
  public long ramBytesUsed() {
    return flatWriter.ramBytesUsed();
  }
}
