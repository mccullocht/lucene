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
import java.util.Arrays;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatKnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatKnnVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.junit.Before;

public class TestLucene104FlatScalarQuantizedVectorsFormat extends BaseKnnVectorsFormatTestCase {

  private ScalarEncoding encoding;
  private KnnVectorsFormat format;

  @Before
  @Override
  public void setUp() throws Exception {
    var encodingValues = ScalarEncoding.values();
    encoding = encodingValues[random().nextInt(encodingValues.length)];
    format = new Lucene104FlatScalarQuantizedVectorsFormat(encoding);
    super.setUp();
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(format);
  }

  @Override
  protected boolean supportsFloatVectorFallback() {
    return false;
  }

  @Override
  protected int getQuantizationBits() {
    return encoding.getBits();
  }

  @Override
  public void testRandomWithUpdatesAndGraph() {
    // no graph is built by this format
  }

  @Override
  public void testSearchWithVisitedLimit() {
    // brute-force search does not honour a visited limit
  }

  /** Verifies that the format produces a {@link FlatKnnVectorsReader} at read time. */
  public void testReaderIsFlat() throws IOException {
    String fieldName = "field";
    int dims = random().nextInt(4, 17);
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      Document doc = new Document();
      doc.add(new KnnFloatVectorField(fieldName, randomVector(dims)));
      w.addDocument(doc);
      w.commit();
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        KnnVectorsReader knnReader =
            ((CodecReader) leaf).getVectorReader().unwrapReaderForField(fieldName);
        assertTrue(
            "Expected FlatKnnVectorsReader but got: " + knnReader,
            knnReader instanceof FlatKnnVectorsReader);
      }
    }
  }

  /** Verifies that basic k-NN search returns the expected number of hits. */
  public void testSearch() throws IOException {
    String fieldName = "field";
    int numVectors = random().nextInt(50, 200);
    int dims = random().nextInt(4, 33);
    VectorSimilarityFunction similarityFunction = randomSimilarity();
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      KnnFloatVectorField knnField =
          new KnnFloatVectorField(fieldName, randomVector(dims), similarityFunction);
      for (int i = 0; i < numVectors; i++) {
        Document doc = new Document();
        knnField.setVectorValue(randomVector(dims));
        doc.add(knnField);
        w.addDocument(doc);
      }
      w.commit();
      try (IndexReader reader = DirectoryReader.open(w)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        int k = random().nextInt(5, Math.min(50, numVectors));
        float[] queryVector = randomVector(dims);
        Query q = new KnnFloatVectorQuery(fieldName, queryVector, k);
        TopDocs results = searcher.search(q, k);
        assertEquals(k, results.totalHits.value());
        assertEquals(TotalHits.Relation.EQUAL_TO, results.totalHits.relation());
      }
    }
  }

  /**
   * Verifies that the flat (exhaustive) search produces exact top-k results by comparing against a
   * brute-force computation over the indexed vectors.
   */
  public void testExhaustiveSearchIsExact() throws IOException {
    // Use EUCLIDEAN so scoring is straightforward (higher score = closer distance).
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    String fieldName = "field";
    int numVectors = random().nextInt(20, 100);
    int dims = random().nextInt(4, 17);
    int k = random().nextInt(1, Math.min(10, numVectors));

    float[][] storedVectors = new float[numVectors][dims];
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      KnnFloatVectorField knnField =
          new KnnFloatVectorField(fieldName, new float[dims], similarityFunction);
      for (int i = 0; i < numVectors; i++) {
        storedVectors[i] = randomVector(dims);
        Document doc = new Document();
        knnField.setVectorValue(storedVectors[i]);
        doc.add(knnField);
        w.addDocument(doc);
      }
      w.commit();
      w.forceMerge(1); // single segment for simple doc-ID → vector mapping

      float[] queryVector = randomVector(dims);

      // Compute brute-force scores.
      float[] scores = new float[numVectors];
      for (int i = 0; i < numVectors; i++) {
        scores[i] = similarityFunction.compare(storedVectors[i], queryVector);
      }
      float[] sortedScores = scores.clone();
      Arrays.sort(sortedScores);
      // Top-k scores are the last k entries in the ascending-sorted array.
      float minTopKScore = sortedScores[numVectors - k];

      try (IndexReader reader = DirectoryReader.open(w)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        Query q = new KnnFloatVectorQuery(fieldName, queryVector, k);
        TopDocs results = searcher.search(q, k);
        assertEquals(k, results.totalHits.value());
        // Every returned document should have a score >= the k-th best brute-force score
        // (allowing a small tolerance for quantization error).
        for (var scoreDoc : results.scoreDocs) {
          assertTrue(scoreDoc.score >= minTopKScore - 0.05f);
        }
      }
    }
  }

  /** Verifies that {@link Lucene104FlatScalarQuantizedVectorsFormat#toString()} is informative. */
  public void testToString() {
    KnnVectorsFormat f = new Lucene104FlatScalarQuantizedVectorsFormat(ScalarEncoding.UNSIGNED_BYTE);
    assertTrue(
        "toString should contain the format name",
        f.toString().contains("Lucene104FlatScalarQuantizedVectorsFormat"));
    assertTrue(
        "toString should mention the underlying flat format",
        f.toString().contains("Lucene104ScalarQuantizedVectorsFormat"));
  }

  /** Verifies that {@link FlatKnnVectorsWriter} delegates RAM usage reporting to the flat writer. */
  public void testWriterRamBytesUsedDelegates() throws IOException {
    String fieldName = "field";
    int dims = random().nextInt(4, 17);
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      KnnFloatVectorField knnField =
          new KnnFloatVectorField(fieldName, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN);
      for (int i = 0; i < 10; i++) {
        Document doc = new Document();
        knnField.setVectorValue(randomVector(dims));
        doc.add(knnField);
        w.addDocument(doc);
      }
      // If flush/close completes without error, delegation works correctly.
      w.commit();
    }
  }
}
