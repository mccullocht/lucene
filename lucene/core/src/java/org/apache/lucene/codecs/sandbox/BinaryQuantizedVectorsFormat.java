package org.apache.lucene.codecs.sandbox;

import java.io.IOException;
import org.apache.lucene.codecs.FlatVectorsFormat;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/** A vector format that stores flat files of full-fidelity and binary quantized vectors. */
public class BinaryQuantizedVectorsFormat extends KnnVectorsFormat {
  private final FlatVectorsFormat flatVectorsFormat = new BinaryQuantizedFlatVectorsFormat();

  public BinaryQuantizedVectorsFormat() {
    super("BinaryQuantizedVectorsFormat");
  }

  static class Writer extends KnnVectorsWriter {
    private final SegmentWriteState state;
    private final FlatVectorsWriter writer;

    Writer(SegmentWriteState state, FlatVectorsWriter writer) {
      this.state = state;
      this.writer = writer;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
      return writer.addField(fieldInfo, null);
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
      writer.flush(maxDoc, sortMap);
    }

    @Override
    public void finish() throws IOException {
      writer.finish();
    }

    @Override
    public void close() throws IOException {
      writer.close();
    }

    @Override
    public long ramBytesUsed() {
      return writer.ramBytesUsed();
    }
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new Writer(state, this.flatVectorsFormat.fieldsWriter(state));
  }

  static class Reader extends KnnVectorsReader {
    private final SegmentReadState state;
    private final FlatVectorsReader reader;

    Reader(SegmentReadState state, FlatVectorsReader reader) {
      this.state = state;
      this.reader = reader;
    }

    @Override
    public void checkIntegrity() throws IOException {
      this.reader.checkIntegrity();
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
      return this.reader.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
      return this.reader.getByteVectorValues(field);
    }

    // NB: we are assuming that if this is cosine similarity that target has been normalized.
    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      RandomVectorScorer scorer = this.reader.getRandomVectorScorer(field, target);
      scanSearch(scorer, knnCollector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      RandomVectorScorer scorer = this.reader.getRandomVectorScorer(field, target);
      scanSearch(scorer, knnCollector, acceptDocs);
    }

    private void scanSearch(RandomVectorScorer scorer, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      // Unfortunately the bits interface is so narrow we can't effectively iterate over the set.
      Bits acceptOrds = scorer.getAcceptOrds(acceptDocs);
      if (acceptOrds == null) {
        for (int ord = 0; ord < scorer.maxOrd(); ord++) {
          knnCollector.collect(scorer.ordToDoc(ord), scorer.score(ord));
        }
      } else {
        for (int ord = 0; ord < scorer.maxOrd(); ord++) {
          if (acceptOrds.get(ord)) {
            knnCollector.collect(scorer.ordToDoc(ord), scorer.score(ord));
          }
        }
      }
    }

    @Override
    public long ramBytesUsed() {
      return this.reader.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
      this.reader.close();
    }
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Reader(state, this.flatVectorsFormat.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 4096;
  }
}
