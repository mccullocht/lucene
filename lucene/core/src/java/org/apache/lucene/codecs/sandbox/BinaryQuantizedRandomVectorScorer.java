package org.apache.lucene.codecs.sandbox;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/** Placate tidy */
public final class BinaryQuantizedRandomVectorScorer implements RandomVectorScorer {
  private final RandomAccessVectorValues<long[]> vectorValues;
  private final long[] quantizedQuery;

  public BinaryQuantizedRandomVectorScorer(
      VectorSimilarityFunction similarityFunction,
      RandomAccessVectorValues<long[]> vectorValues,
      float[] query) {
    this(vectorValues, quantizeQuery(similarityFunction, query));
  }

  private static long[] quantizeQuery(VectorSimilarityFunction similarityFunction, float[] query) {
    float[] processedQuery =
        switch (similarityFunction) {
          case COSINE -> {
            float[] queryCopy = ArrayUtil.copyOfSubArray(query, 0, query.length);
            VectorUtil.l2normalize(queryCopy);
            yield queryCopy;
          }
          case EUCLIDEAN, DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> query;
        };
    return BinaryQuantizationUtils.quantize(processedQuery);
  }

  public BinaryQuantizedRandomVectorScorer(
      RandomAccessVectorValues<long[]> vectorValues, byte[] query) {
    this(vectorValues, BinaryQuantizationUtils.quantize(query));
  }

  BinaryQuantizedRandomVectorScorer(RandomAccessVectorValues<long[]> vectorValues, long[] query) {
    this.vectorValues = vectorValues;
    this.quantizedQuery = query;
  }

  @Override
  public float score(int node) throws IOException {
    return BinaryQuantizationUtils.score(this.quantizedQuery, this.vectorValues.vectorValue(node));
  }

  @Override
  public int maxOrd() {
    return this.vectorValues.size();
  }

  @Override
  public int ordToDoc(int ord) {
    return this.vectorValues.ordToDoc(ord);
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    return this.vectorValues.getAcceptOrds(acceptDocs);
  }
}
