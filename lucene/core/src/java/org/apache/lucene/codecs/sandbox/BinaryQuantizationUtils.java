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
    int distance = 0;
    for (int i = 0; i < a.length; i++) {
      distance += Long.bitCount(a[i] ^ b[i]);
    }
    int dim = a.length * 64;
    return (float) (dim - distance) / dim;
  }

  public static float score(long[] a, long[] b, float minScore) {
    assert a.length == b.length;
    int dim = a.length * 64;
    final int maxDistance = dim - (int) (minScore * dim);
    int distance = 0;
    for (int i = 0; i < a.length; i += 2) {
      distance += Long.bitCount(a[i] ^ b[i]) + Long.bitCount(a[i + 1] ^ b[i + 1]);
      if (distance > maxDistance) {
        return 0.0f;
      }
    }
    return (float) (dim - distance) / dim;
  }
}
