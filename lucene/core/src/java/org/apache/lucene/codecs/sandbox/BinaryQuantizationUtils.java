package org.apache.lucene.codecs.sandbox;

import org.apache.lucene.index.VectorSimilarityFunction;

/** Placate tidy. */
public final class BinaryQuantizationUtils {
  private BinaryQuantizationUtils() {}

  public static int byteSize(int dimensions) {
    // Round up to 16 bytes to improve ILP/SIMD possibilities.
    return (dimensions + 127) / 8;
  }

  public static long[] allocate(int dimensions) {
    return new long[byteSize(dimensions) / Long.BYTES];
  }

  public static void quantize(float[] vector, long[] binVector) {
    for (int i = 0; i < vector.length; i++) {
      if (vector[i] > 0.0) {
        binVector[i / 64] |= 1L << (i % 64);
      }
    }
  }

  public static long[] quantize(float[] vector) {
    var binVector = allocate(vector.length);
    quantize(vector, binVector);
    return binVector;
  }

  public static void quantize(byte[] vector, long[] binVector) {
    for (int i = 0; i < vector.length; i++) {
      if (vector[i] >= 0) {
        binVector[i / 64] |= 1L << (i % 64);
      }
    }
  }

  public static long[] quantize(byte[] vector) {
    var binVector = allocate(vector.length);
    quantize(vector, binVector);
    return binVector;
  }

  public static float score(
      long[] vector1, long[] vector2, int dimensions, VectorSimilarityFunction function) {
    if (function == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
      throw new UnsupportedOperationException();
    }

    assert vector1.length == vector2.length;
    long xor_popcnt = 0;
    for (int i = 0; i < vector1.length; i++) {
      xor_popcnt += Long.bitCount(vector1[i] ^ vector2[i]);
    }
    long unset = dimensions - xor_popcnt;
    return (float) (xor_popcnt - unset);
  }
}
