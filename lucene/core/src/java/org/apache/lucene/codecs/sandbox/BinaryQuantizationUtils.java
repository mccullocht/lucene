package org.apache.lucene.codecs.sandbox;

import java.util.Arrays;
import org.apache.lucene.index.VectorSimilarityFunction;

/** Placate tidy. */
public final class BinaryQuantizationUtils {
  private static final float[] UQ_FLOAT_VALUES = new float[] {-1.0f, 1.0f};
  private static final byte[] UQ_BYTE_VALUES = new byte[] {-1, 1};

  // For any byte of input binary vector the index at byte * 8 contains the 8 float values that
  // would be decoded from it as part of un-quantization. This allows un-quantization to decode 8
  // values at a time via memcpy instead of one bit at a time.
  private static final float[] UQ_FLOAT_DECODE_TABLE;

  static {
    UQ_FLOAT_DECODE_TABLE = new float[256 * 8];
    for (int b = 0; b < 256; b++) {
      for (int i = 0; i < 8; i++) {
        UQ_FLOAT_DECODE_TABLE[b * 8 + i] = UQ_FLOAT_VALUES[(b >> i) & 0x1];
      }
    }
  }

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

  private static void unQuantizeByte(long b, float[] vector, int offset) {
    System.arraycopy(UQ_FLOAT_DECODE_TABLE, (int) b * 8, vector, offset, 8);
  }

  public static void unQuantize(long[] binVector, float[] vector) {
    Arrays.fill(vector, -1.0f);
    for (int i = 0; i < binVector.length; i++) {
      long d64 = binVector[i];
      while (d64 != 0) {
        int setBit = Long.numberOfTrailingZeros(d64);
        vector[i * 64 + setBit] = 1.0f;
        d64 ^= (1 << setBit);
      }
    }
    /*
    for (int i = 0; i < binVector.length; i++) {
      long d64 = binVector[i];
      unQuantizeByte(d64 & 0xff, vector, i * 64);
      unQuantizeByte((d64 >> 8) & 0xff, vector, i * 64 + 8);
      unQuantizeByte((d64 >> 16) & 0xff, vector, i * 64 + 16);
      unQuantizeByte((d64 >> 24) & 0xff, vector, i * 64 + 24);
      unQuantizeByte((d64 >> 32) & 0xff, vector, i * 64 + 32);
      unQuantizeByte((d64 >> 40) & 0xff, vector, i * 64 + 40);
      unQuantizeByte((d64 >> 48) & 0xff, vector, i * 64 + 48);
      unQuantizeByte((d64 >> 56) & 0xff, vector, i * 64 + 56);
    }
     */
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
        vector[i * 64 + j] = UQ_BYTE_VALUES[(int) (d64 >> j) & 0x1];
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
