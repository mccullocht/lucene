package org.apache.lucene.codecs.sandbox;

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

  public static float score(long[] a, long[] b) {
    assert a.length == b.length;
    int i = 0;
    long count0 = 0;
    long count1 = 0;
    long count2 = 0;
    long count3 = 0;
    int limit = a.length & ~0x3;
    for (; i < limit; i += 4) {
      count0 += Long.bitCount(a[i] ^ b[i]);
      count1 += Long.bitCount(a[i + 1] ^ b[i + 1]);
      count2 += Long.bitCount(a[i + 2] ^ b[i + 2]);
      count3 += Long.bitCount(a[i + 3] ^ b[i + 3]);
    }
    for (; i < a.length; i++) {
      count0 += Long.bitCount(a[i] ^ b[i]);
    }
    return 1.0f / (1.0f + count0 + count1 + count2 + count3);
  }
}
