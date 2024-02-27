package org.apache.lucene.codecs.sandbox;

import java.io.IOException;

/** Placate tidy. */
public interface BinaryQuantizedVectorsReader {
  BinaryVectorValues getBinaryVectorValues(String fieldName) throws IOException;
}
