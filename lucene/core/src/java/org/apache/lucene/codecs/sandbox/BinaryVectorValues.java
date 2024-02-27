package org.apache.lucene.codecs.sandbox;

import java.io.IOException;
import org.apache.lucene.search.DocIdSetIterator;

/** Provides access to binary-coded vector values (presented as long[]). */
public abstract class BinaryVectorValues extends DocIdSetIterator {
  /** Sole constructor */
  protected BinaryVectorValues() {}

  /** Return the dimension of the vectors */
  public abstract int dimension();

  /**
   * Return the number of vectors for this field.
   *
   * @return the number of vectors returned by this iterator
   */
  public abstract int size();

  @Override
  public final long cost() {
    return size();
  }

  /**
   * Return the vector value for the current document ID. It is illegal to call this method when the
   * iterator is not positioned: before advancing, or after failing to advance. The returned array
   * may be shared across calls, re-used, and modified as the iterator advances.
   *
   * @return the vector value
   */
  public abstract long[] vectorValue() throws IOException;
}
