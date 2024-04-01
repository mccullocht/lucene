/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.sandbox;

import static org.apache.lucene.codecs.sandbox.BinaryQuantizedFlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.FlatVectorsWriter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/**
 * Writes quantized vector values and metadata to index segments.
 *
 * @lucene.experimental
 */
public final class BinaryQuantizedFlatVectorsWriter extends FlatVectorsWriter {
  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(BinaryQuantizedFlatVectorsWriter.class);
  private static final int VECTOR_ALIGNMENT = 16;

  private final SegmentWriteState segmentWriteState;

  private final List<FieldWriter<?>> fields = new ArrayList<>();
  private final IndexOutput meta, quantizedVectorData;
  private boolean finished;

  public BinaryQuantizedFlatVectorsWriter(SegmentWriteState state) throws IOException {
    segmentWriteState = state;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            BinaryQuantizedFlatVectorsFormat.META_EXTENSION);

    String quantizedVectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            BinaryQuantizedFlatVectorsFormat.VECTOR_DATA_EXTENSION);
    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      quantizedVectorData =
          state.directory.createOutput(quantizedVectorDataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          BinaryQuantizedFlatVectorsFormat.META_CODEC_NAME,
          BinaryQuantizedFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          quantizedVectorData,
          BinaryQuantizedFlatVectorsFormat.VECTOR_DATA_CODEC_NAME,
          BinaryQuantizedFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  @Override
  public FlatFieldVectorsWriter<?> addField(
      FieldInfo fieldInfo, KnnFieldVectorsWriter<?> indexWriter) throws IOException {
    FieldWriter<?> quantizedWriter = FieldWriter.create(fieldInfo, indexWriter);
    this.fields.add(quantizedWriter);
    return quantizedWriter;
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    // Since we know we will not be searching for additional indexing, we can just write the
    // the vectors directly to the new segment.
    // No need to use temporary file as we don't have to re-open for reading
    MergedQuantizedVectorValues binaryVectorValues =
        MergedQuantizedVectorValues.mergeQuantizedBinaryVectorValues(fieldInfo, mergeState);
    long vectorDataOffset = quantizedVectorData.alignFilePointer(VECTOR_ALIGNMENT);
    DocsWithFieldSet docsWithField =
        writeQuantizedVectorData(quantizedVectorData, binaryVectorValues);
    long vectorDataLength = quantizedVectorData.getFilePointer() - vectorDataOffset;
    writeMeta(
        fieldInfo,
        segmentWriteState.segmentInfo.maxDoc(),
        vectorDataOffset,
        vectorDataLength,
        docsWithField);
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    long vectorDataOffset = quantizedVectorData.alignFilePointer(VECTOR_ALIGNMENT);
    IndexOutput tempQuantizedVectorData =
        segmentWriteState.directory.createTempOutput(
            quantizedVectorData.getName(), "temp", segmentWriteState.context);
    IndexInput quantizationDataInput = null;
    boolean success = false;
    try {
      MergedQuantizedVectorValues binaryVectorValues =
          MergedQuantizedVectorValues.mergeQuantizedBinaryVectorValues(fieldInfo, mergeState);
      DocsWithFieldSet docsWithField =
          writeQuantizedVectorData(tempQuantizedVectorData, binaryVectorValues);
      CodecUtil.writeFooter(tempQuantizedVectorData);
      IOUtils.close(tempQuantizedVectorData);
      quantizationDataInput =
          segmentWriteState.directory.openInput(
              tempQuantizedVectorData.getName(), segmentWriteState.context);
      quantizedVectorData.copyBytes(
          quantizationDataInput, quantizationDataInput.length() - CodecUtil.footerLength());
      long vectorDataLength = quantizedVectorData.getFilePointer() - vectorDataOffset;
      CodecUtil.retrieveChecksum(quantizationDataInput);
      writeMeta(
          fieldInfo,
          segmentWriteState.segmentInfo.maxDoc(),
          vectorDataOffset,
          vectorDataLength,
          docsWithField);
      success = true;
      final IndexInput finalQuantizationDataInput = quantizationDataInput;
      return new BinaryQuantizedCloseableRandomVectorScorerSupplier(
          () -> {
            IOUtils.close(finalQuantizationDataInput);
            segmentWriteState.directory.deleteFile(tempQuantizedVectorData.getName());
          },
          docsWithField.cardinality(),
          new BinaryQuantizedRandomVectorScorerSupplier(
              new OffHeapQuantizedBinaryVectorValues.DenseOffHeapVectorValues(
                  fieldInfo.getVectorDimension(),
                  docsWithField.cardinality(),
                  quantizationDataInput)));
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(tempQuantizedVectorData, quantizationDataInput);
        IOUtils.deleteFilesIgnoringExceptions(
            segmentWriteState.directory, tempQuantizedVectorData.getName());
      }
    }
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (FieldWriter<?> field : fields) {
      field.finish();
      if (sortMap == null) {
        writeField(field, maxDoc);
      } else {
        writeSortingField(field, maxDoc, sortMap);
      }
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (quantizedVectorData != null) {
      CodecUtil.writeFooter(quantizedVectorData);
    }
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (FieldWriter<?> field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }

  private void writeField(FieldWriter<?> fieldData, int maxDoc) throws IOException {
    // write vector values
    long vectorDataOffset = quantizedVectorData.alignFilePointer(VECTOR_ALIGNMENT);
    for (long[] v : fieldData.binaryVectors) {
      writeQuantizedVector(v);
    }
    long vectorDataLength = quantizedVectorData.getFilePointer() - vectorDataOffset;

    writeMeta(
        fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, fieldData.docsWithField);
  }

  private void writeSortingField(FieldWriter<?> fieldData, int maxDoc, Sorter.DocMap sortMap)
      throws IOException {
    final int[] docIdOffsets = new int[sortMap.size()];
    int offset = 1; // 0 means no vector for this (field, document)
    DocIdSetIterator iterator = fieldData.docsWithField.iterator();
    for (int docID = iterator.nextDoc();
        docID != DocIdSetIterator.NO_MORE_DOCS;
        docID = iterator.nextDoc()) {
      int newDocID = sortMap.oldToNew(docID);
      docIdOffsets[newDocID] = offset++;
    }
    DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
    final int[] ordMap = new int[offset - 1]; // new ord to old ord
    int ord = 0;
    int doc = 0;
    for (int docIdOffset : docIdOffsets) {
      if (docIdOffset != 0) {
        ordMap[ord] = docIdOffset - 1;
        newDocsWithField.add(doc);
        ord++;
      }
      doc++;
    }

    // write vector values
    long vectorDataOffset = quantizedVectorData.alignFilePointer(VECTOR_ALIGNMENT);
    for (int ordinal : ordMap) {
      writeQuantizedVector(fieldData.binaryVectors.get(ordinal));
    }
    long quantizedVectorLength = quantizedVectorData.getFilePointer() - vectorDataOffset;
    writeMeta(
        fieldData.fieldInfo, maxDoc, vectorDataOffset, quantizedVectorLength, newDocsWithField);
  }

  private void writeQuantizedVector(long[] vectorValue) throws IOException {
    writeQuantizedVector(vectorValue, this.quantizedVectorData);
  }

  private static void writeQuantizedVector(long[] vectorValue, IndexOutput output)
      throws IOException {
    for (long bits : vectorValue) {
      output.writeLong(bits);
    }
  }

  private void writeMeta(
      FieldInfo field,
      int maxDoc,
      long vectorDataOffset,
      long vectorDataLength,
      DocsWithFieldSet docsWithField)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    meta.writeVInt(field.getVectorDimension());
    int count = docsWithField.cardinality();
    meta.writeInt(count);
    // write docIDs
    OrdToDocDISIReaderConfiguration.writeStoredMeta(
        DIRECT_MONOTONIC_BLOCK_SHIFT, meta, quantizedVectorData, count, maxDoc, docsWithField);
  }

  private static BinaryQuantizedVectorsReader getQuantizedKnnVectorsReader(
      KnnVectorsReader vectorsReader, String fieldName) {
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
      vectorsReader = candidateReader.getFieldReader(fieldName);
    }
    if (vectorsReader instanceof BinaryQuantizedVectorsReader reader) {
      return reader;
    }
    return null;
  }

  /**
   * Writes the vector values to the output and returns a set of documents that contains vectors.
   */
  public static DocsWithFieldSet writeQuantizedVectorData(
      IndexOutput output, BinaryVectorValues binaryVectorValues) throws IOException {
    DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    for (int docV = binaryVectorValues.nextDoc();
        docV != NO_MORE_DOCS;
        docV = binaryVectorValues.nextDoc()) {
      // write vector
      long[] binaryValue = binaryVectorValues.vectorValue();
      assert binaryValue.length
              == BinaryQuantizationUtils.byteSize(binaryVectorValues.dimension()) / Long.BYTES
          : "dim=" + binaryVectorValues.dimension() + " len=" + binaryValue.length;
      writeQuantizedVector(binaryValue, output);
      docsWithField.add(docV);
    }
    return docsWithField;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, quantizedVectorData);
  }

  abstract static class FieldWriter<T> extends FlatFieldVectorsWriter<T> {
    private static final long SHALLOW_SIZE = shallowSizeOfInstance(FieldWriter.class);
    private final List<long[]> binaryVectors;
    private final FieldInfo fieldInfo;
    private final DocsWithFieldSet docsWithField;

    @SuppressWarnings("unchecked")
    FieldWriter(FieldInfo fieldInfo, KnnFieldVectorsWriter<T> indexWriter) {
      super(indexWriter);
      this.fieldInfo = fieldInfo;
      this.binaryVectors = new ArrayList<>();
      this.docsWithField = new DocsWithFieldSet();
    }

    @SuppressWarnings("unchecked")
    static FieldWriter<?> create(FieldInfo fieldInfo, KnnFieldVectorsWriter<?> indexWriter) {
      boolean normalize =
          fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE;
      return switch (fieldInfo.getVectorEncoding()) {
        case BYTE -> new FieldWriter<>(fieldInfo, (KnnFieldVectorsWriter<byte[]>) indexWriter) {
          @Override
          protected long[] quantizeVector(byte[] vectorValue) {
            return BinaryQuantizationUtils.quantize(vectorValue);
          }
        };
        case FLOAT32 -> new FieldWriter<>(fieldInfo, (KnnFieldVectorsWriter<float[]>) indexWriter) {
          float[] copy = normalize ? new float[fieldInfo.getVectorDimension()] : null;

          @Override
          protected long[] quantizeVector(float[] vectorValue) {
            if (normalize) {
              System.arraycopy(vectorValue, 0, copy, 0, copy.length);
              VectorUtil.l2normalize(copy);
              vectorValue = copy;
            }
            return BinaryQuantizationUtils.quantize(vectorValue);
          }
        };
      };
    }

    protected abstract long[] quantizeVector(T vectorValue);

    void finish() throws IOException {}

    @Override
    public long ramBytesUsed() {
      long size = SHALLOW_SIZE;
      if (indexingDelegate != null) {
        size += indexingDelegate.ramBytesUsed();
      }
      if (this.binaryVectors.size() == 0) return size;
      return size + (long) this.binaryVectors.size() * RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    }

    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
      docsWithField.add(docID);
      this.binaryVectors.add(quantizeVector(vectorValue));
      // NB: the call chain is flatFormat => quantized format (here) => hnsw (optional)
      // We are passing the original value so the graph will be built on the original distances and
      // not quantized. It is likely we will eventually rebuild the graph based on quantized values
      // but it is also not guaranteed to happen, especially if there are no deletions.
      if (indexingDelegate != null) {
        indexingDelegate.addValue(docID, vectorValue);
      }
    }

    @Override
    public T copyValue(T vectorValue) {
      throw new UnsupportedOperationException();
    }
  }

  private static class QuantizedByteVectorValueSub extends DocIDMerger.Sub {
    private final BinaryVectorValues values;

    QuantizedByteVectorValueSub(MergeState.DocMap docMap, BinaryVectorValues values) {
      super(docMap);
      this.values = values;
      assert values.docID() == -1;
    }

    @Override
    public int nextDoc() throws IOException {
      return values.nextDoc();
    }
  }

  /** Returns a merged view over all the segment's {@link BinaryVectorValues}. */
  static class MergedQuantizedVectorValues extends BinaryVectorValues {
    public static MergedQuantizedVectorValues mergeQuantizedBinaryVectorValues(
        FieldInfo fieldInfo, MergeState mergeState) throws IOException {
      assert fieldInfo != null && fieldInfo.hasVectorValues();

      List<QuantizedByteVectorValueSub> subs = new ArrayList<>();
      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        if (mergeState.knnVectorsReaders[i] != null
            && mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name) != null) {
          BinaryQuantizedVectorsReader reader =
              getQuantizedKnnVectorsReader(mergeState.knnVectorsReaders[i], fieldInfo.name);
          final BinaryVectorValues values;
          if (reader == null) {
            // The input may not have been quantized yet; do this transform now.
            values =
                switch (fieldInfo.getVectorEncoding()) {
                  case BYTE -> new BinaryQuantizedFlatVectorsWriter.QuantizedByteVectorValues(
                      mergeState.knnVectorsReaders[i].getByteVectorValues(fieldInfo.name));
                  case FLOAT32 -> new QuantizedFloatVectorValues(
                      mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldInfo.name),
                      fieldInfo.getVectorSimilarityFunction());
                };
          } else {
            values = reader.getBinaryVectorValues(fieldInfo.name);
          }
          subs.add(new QuantizedByteVectorValueSub(mergeState.docMaps[i], values));
        }
      }
      return new MergedQuantizedVectorValues(subs, mergeState);
    }

    private final List<QuantizedByteVectorValueSub> subs;
    private final DocIDMerger<QuantizedByteVectorValueSub> docIdMerger;
    private final int size;

    private int docId;
    private QuantizedByteVectorValueSub current;

    private MergedQuantizedVectorValues(
        List<QuantizedByteVectorValueSub> subs, MergeState mergeState) throws IOException {
      this.subs = subs;
      docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
      int totalSize = 0;
      for (QuantizedByteVectorValueSub sub : subs) {
        totalSize += sub.values.size();
      }
      size = totalSize;
      docId = -1;
    }

    @Override
    public long[] vectorValue() throws IOException {
      return current.values.vectorValue();
    }

    @Override
    public int docID() {
      return docId;
    }

    @Override
    public int nextDoc() throws IOException {
      current = docIdMerger.next();
      if (current == null) {
        docId = NO_MORE_DOCS;
      } else {
        docId = current.mappedDocID;
      }
      return docId;
    }

    @Override
    public int advance(int target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int dimension() {
      return subs.get(0).values.dimension();
    }
  }

  private static class QuantizedFloatVectorValues extends BinaryVectorValues {
    private final FloatVectorValues values;
    private final long[] quantizedVector;
    private final float[] normalizedVector;

    private final VectorSimilarityFunction vectorSimilarityFunction;

    public QuantizedFloatVectorValues(
        FloatVectorValues values, VectorSimilarityFunction vectorSimilarityFunction) {
      this.values = values;
      this.quantizedVector = BinaryQuantizationUtils.allocate(values.dimension());
      this.vectorSimilarityFunction = vectorSimilarityFunction;
      if (vectorSimilarityFunction == VectorSimilarityFunction.COSINE) {
        this.normalizedVector = new float[values.dimension()];
      } else {
        this.normalizedVector = null;
      }
    }

    @Override
    public int dimension() {
      return values.dimension();
    }

    @Override
    public int size() {
      return values.size();
    }

    @Override
    public long[] vectorValue() throws IOException {
      return quantizedVector;
    }

    @Override
    public int docID() {
      return values.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      int doc = values.nextDoc();
      if (doc != NO_MORE_DOCS) {
        quantize();
      }
      return doc;
    }

    @Override
    public int advance(int target) throws IOException {
      int doc = values.advance(target);
      if (doc != NO_MORE_DOCS) {
        quantize();
      }
      return doc;
    }

    private void quantize() throws IOException {
      if (vectorSimilarityFunction == VectorSimilarityFunction.COSINE) {
        System.arraycopy(values.vectorValue(), 0, normalizedVector, 0, normalizedVector.length);
        VectorUtil.l2normalize(normalizedVector);
        BinaryQuantizationUtils.quantize(normalizedVector, quantizedVector);
      } else {
        BinaryQuantizationUtils.quantize(values.vectorValue(), quantizedVector);
      }
    }
  }

  private static class QuantizedByteVectorValues extends BinaryVectorValues {
    private final ByteVectorValues values;
    private final long[] quantizedVector;

    public QuantizedByteVectorValues(ByteVectorValues values) {
      this.values = values;
      this.quantizedVector = BinaryQuantizationUtils.allocate(values.dimension());
    }

    @Override
    public int dimension() {
      return values.dimension();
    }

    @Override
    public int size() {
      return values.size();
    }

    @Override
    public long[] vectorValue() throws IOException {
      return quantizedVector;
    }

    @Override
    public int docID() {
      return values.docID();
    }

    @Override
    public int nextDoc() throws IOException {
      int doc = values.nextDoc();
      if (doc != NO_MORE_DOCS) {
        quantize();
      }
      return doc;
    }

    @Override
    public int advance(int target) throws IOException {
      int doc = values.advance(target);
      if (doc != NO_MORE_DOCS) {
        quantize();
      }
      return doc;
    }

    private void quantize() throws IOException {
      BinaryQuantizationUtils.quantize(values.vectorValue(), quantizedVector);
    }
  }

  static class BinaryQuantizedRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
    private final RandomAccessVectorValues<long[]> values;

    public BinaryQuantizedRandomVectorScorerSupplier(RandomAccessVectorValues<long[]> values) {
      this.values = values;
    }

    @Override
    public RandomVectorScorer scorer(int ord) throws IOException {
      RandomAccessVectorValues<long[]> valuesCopy = values.copy();
      // NB: we are implicitly relying on the notion that only one scorer will be used at a time,
      // because a subsequent call to scorer() will overwrite the contents of queryVector.
      long[] queryVector = this.values.vectorValue(ord);
      return new BinaryQuantizedRandomVectorScorer(valuesCopy, queryVector);
    }

    @Override
    public RandomVectorScorerSupplier copy() throws IOException {
      return new BinaryQuantizedRandomVectorScorerSupplier(this.values.copy());
    }
  }

  static final class BinaryQuantizedCloseableRandomVectorScorerSupplier
      implements CloseableRandomVectorScorerSupplier {

    private final BinaryQuantizedRandomVectorScorerSupplier supplier;
    private final Closeable onClose;
    private final int numVectors;

    BinaryQuantizedCloseableRandomVectorScorerSupplier(
        Closeable onClose, int numVectors, BinaryQuantizedRandomVectorScorerSupplier supplier) {
      this.onClose = onClose;
      this.supplier = supplier;
      this.numVectors = numVectors;
    }

    @Override
    public RandomVectorScorer scorer(int ord) throws IOException {
      return supplier.scorer(ord);
    }

    @Override
    public RandomVectorScorerSupplier copy() throws IOException {
      return supplier.copy();
    }

    @Override
    public void close() throws IOException {
      onClose.close();
    }

    @Override
    public int totalVectorCount() {
      return numVectors;
    }
  }
}
