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
    // TODO: test scoring byte array representation, maybe it will behave better.
    long count = 0;
    for (int i = 0; i < a.length; i++) {
      count += Long.bitCount(a[i] ^ b[i]);
    }
    return 1.0f / (1.0f + count);
  }

  static {
    System.loadLibrary("hammer");
  }

  public static native int distance(long aAddr, long bAddr, int len);
}
