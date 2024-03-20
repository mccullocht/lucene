package org.apache.lucene.codecs.sandbox;

import org.apache.lucene.index.VectorSimilarityFunction;

/** Placate tidy. */
public final class BinaryQuantizationUtils {
  private BinaryQuantizationUtils() {}

  public static int byteSize(int dimensions) {
    // Round up to 16 bytes to improve ILP/SIMD possibilities.
    int quads = (dimensions + 127) / 128;
    return quads * 16;
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

  public static void unQuantize(long[] binVector, float[] vector) {
    for (int i = 0; i < binVector.length; i++) {
      long d64 = binVector[i];
      for (int j = 0; j < 64; j++) {
        vector[i * 64 + j] = (d64 & 1L << j) == 1 ? 1.0f : -1.0f;
      }
    }
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

  public static void unQuantize(long[] binVector, byte[] vector) {
    for (int i = 0; i < binVector.length; i++) {
      long d64 = binVector[i];
      for (int j = 0; j < 64; j++) {
        vector[i * 64 + j] = (d64 & 1L << j) == 1 ? (byte) 1 : (byte) -1;
      }
    }
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
    return 1.0f / (1.0f + xor_popcnt);
  }
}
