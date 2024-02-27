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

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FlatVectorsReader;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Read binary quantized flat vectors in parallel with byte/float vectors.
 *
 * @lucene.experimental
 */
public final class BinaryQuantizedFlatVectorsReader extends FlatVectorsReader
    implements BinaryQuantizedVectorsReader {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(BinaryQuantizedFlatVectorsReader.class);

  private final Map<String, FieldEntry> fields = new HashMap<>();
  private final IndexInput quantizedVectorData;
  private final FlatVectorsReader rawVectorsReader;

  public BinaryQuantizedFlatVectorsReader(
      SegmentReadState state, FlatVectorsReader rawVectorsReader) throws IOException {
    this.rawVectorsReader = rawVectorsReader;
    int versionMeta = -1;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            BinaryQuantizedFlatVectorsFormat.META_EXTENSION);
    boolean success = false;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                BinaryQuantizedFlatVectorsFormat.META_CODEC_NAME,
                BinaryQuantizedFlatVectorsFormat.VERSION_START,
                BinaryQuantizedFlatVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(meta, state.fieldInfos);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
      quantizedVectorData =
          openDataInput(
              state,
              versionMeta,
              BinaryQuantizedFlatVectorsFormat.VECTOR_DATA_EXTENSION,
              BinaryQuantizedFlatVectorsFormat.VECTOR_DATA_CODEC_NAME);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.name, fieldEntry);
    }
  }

  static void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dimension) {
      throw new IllegalStateException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + fieldEntry.dimension);
    }

    long quantizedVectorBytes = BinaryQuantizationUtils.byteSize(dimension);
    long numQuantizedVectorBytes = Math.multiplyExact(quantizedVectorBytes, fieldEntry.size);
    if (numQuantizedVectorBytes != fieldEntry.vectorDataLength) {
      throw new IllegalStateException(
          "Quantized vector data length "
              + fieldEntry.vectorDataLength
              + " not matching size="
              + fieldEntry.size
              + " * (dim=("
              + dimension
              + " + 127) / 8)"
              + " = "
              + numQuantizedVectorBytes);
    }
  }

  @Override
  public void checkIntegrity() throws IOException {
    rawVectorsReader.checkIntegrity();
    CodecUtil.checksumEntireFile(quantizedVectorData);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return rawVectorsReader.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return rawVectorsReader.getByteVectorValues(field);
  }

  private static IndexInput openDataInput(
      SegmentReadState state, int versionMeta, String fileExtension, String codecName)
      throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    IndexInput in = state.directory.openInput(fileName, state.context);
    boolean success = false;
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              codecName,
              BinaryQuantizedFlatVectorsFormat.VERSION_START,
              BinaryQuantizedFlatVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta="
                + versionMeta
                + ", "
                + codecName
                + "="
                + versionVectorData,
            in);
      }
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String fieldName, float[] target)
      throws IOException {
    FieldEntry field = this.fields.get(fieldName);
    var vectorValues = loadVectorValues(field);
    if (vectorValues == null) {
      return null;
    }
    return new BinaryQuantizedRandomVectorScorer(field.similarityFunction, vectorValues, target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String fieldName, byte[] target)
      throws IOException {
    FieldEntry field = this.fields.get(fieldName);
    var vectorValues = loadVectorValues(field);
    if (vectorValues == null) {
      return null;
    }
    return new BinaryQuantizedRandomVectorScorer(field.similarityFunction, vectorValues, target);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(quantizedVectorData, rawVectorsReader);
  }

  @Override
  public long ramBytesUsed() {
    long size = SHALLOW_SIZE;
    size +=
        RamUsageEstimator.sizeOfMap(
            fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class));
    size += rawVectorsReader.ramBytesUsed();
    return size;
  }

  private FieldEntry readField(IndexInput input) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(input);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
      throw new UnsupportedOperationException();
    }
    return new FieldEntry(input, vectorEncoding, similarityFunction);
  }

  @Override
  public BinaryVectorValues getBinaryVectorValues(String fieldName) throws IOException {
    return loadVectorValues(this.fields.get(fieldName));
  }

  OffHeapQuantizedBinaryVectorValues loadVectorValues(FieldEntry field) throws IOException {
    if (field == null) {
      return null;
    }
    return OffHeapQuantizedBinaryVectorValues.load(
        field.ordToDoc,
        field.dimension,
        field.size,
        field.vectorDataOffset,
        field.vectorDataLength,
        quantizedVectorData);
  }

  private static class FieldEntry implements Accountable {
    private static final long SHALLOW_SIZE =
        RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class);
    final VectorSimilarityFunction similarityFunction;
    final VectorEncoding vectorEncoding;
    final int dimension;
    final long vectorDataOffset;
    final long vectorDataLength;
    final int size;
    final OrdToDocDISIReaderConfiguration ordToDoc;

    FieldEntry(
        IndexInput input,
        VectorEncoding vectorEncoding,
        VectorSimilarityFunction similarityFunction)
        throws IOException {
      this.similarityFunction = similarityFunction;
      this.vectorEncoding = vectorEncoding;
      vectorDataOffset = input.readVLong();
      vectorDataLength = input.readVLong();
      dimension = input.readVInt();
      size = input.readInt();
      ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(input, size);
    }

    @Override
    public long ramBytesUsed() {
      return SHALLOW_SIZE + RamUsageEstimator.sizeOf(ordToDoc);
    }
  }
}
