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
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * A {@link KnnVectorsReader} that delegates storage to an underlying {@link FlatVectorsReader} and
 * performs exhaustive nearest-neighbor search over all stored vectors. Each {@link #search} call
 * scores every vector in the index in batches, respecting {@link AcceptDocs} filtering and {@link
 * KnnCollector#earlyTerminated()} for early exit.
 *
 * @lucene.experimental
 */
public final class FlatKnnVectorsReader extends KnnVectorsReader {

  private static final int BULK_SCORE_ORDS = 64;

  private final FlatVectorsReader flatReader;

  /** Creates a new reader delegating to the given flat reader. */
  public FlatKnnVectorsReader(FlatVectorsReader flatReader) {
    this.flatReader = flatReader;
  }

  @Override
  public void checkIntegrity() throws IOException {
    flatReader.checkIntegrity();
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return flatReader.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return flatReader.getByteVectorValues(field);
  }

  @Override
  public void search(
      String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    if (knnCollector.k() == 0) return;
    RandomVectorScorer scorer = flatReader.getRandomVectorScorer(field, target);
    if (scorer == null) return;
    exhaustiveSearch(scorer, knnCollector, acceptDocs);
  }

  @Override
  public void search(
      String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    if (knnCollector.k() == 0) return;
    RandomVectorScorer scorer = flatReader.getRandomVectorScorer(field, target);
    if (scorer == null) return;
    exhaustiveSearch(scorer, knnCollector, acceptDocs);
  }

  private static void exhaustiveSearch(
      RandomVectorScorer scorer, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs.bits());
    int[] ords = new int[BULK_SCORE_ORDS];
    float[] scores = new float[BULK_SCORE_ORDS];
    int numOrds = 0;
    int numVectors = scorer.maxOrd();
    for (int i = 0; i < numVectors; i++) {
      if (acceptedOrds == null || acceptedOrds.get(i)) {
        if (knnCollector.earlyTerminated()) {
          break;
        }
        ords[numOrds++] = i;
        if (numOrds == ords.length) {
          knnCollector.incVisitedCount(numOrds);
          if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
            for (int j = 0; j < numOrds; j++) {
              knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
            }
          }
          numOrds = 0;
        }
      }
    }
    if (numOrds > 0) {
      knnCollector.incVisitedCount(numOrds);
      if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
        for (int j = 0; j < numOrds; j++) {
          knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
        }
      }
    }
  }

  @Override
  public KnnVectorsReader getMergeInstance() throws IOException {
    return new FlatKnnVectorsReader(flatReader.getMergeInstance());
  }

  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    return flatReader.getOffHeapByteSize(fieldInfo);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(flatReader);
  }
}
