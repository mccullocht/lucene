package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.codecs.sandbox.BinaryQuantizedVectorsReader;
import org.apache.lucene.codecs.sandbox.BinaryVectorValues;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.VectorSimilarityFunction;

/** I'm just here so the build doesn't fail. */
public class KnnFloatRescoreVectorQuery extends KnnFloatVectorQuery {
  private static final float OVERSAMPLE;

  // XXX VIOLENCE
  private static final TaskExecutor TASK_EXECUTOR =
      new TaskExecutor(new ScheduledThreadPoolExecutor(16));

  static {
    float oversample;
    try {
      oversample = Float.parseFloat(System.getenv("BQ_SEGMENT_RESCORE_OVERSAMPLE"));
    } catch (NullPointerException | NumberFormatException e) {
      oversample = 1.0f;
    }
    OVERSAMPLE = oversample;
  }

  // NB: the superclass has this as a private member.
  private final float[] target;
  // NB: captured to access segment state to re-score.
  private IndexSearcher searcher;

  public KnnFloatRescoreVectorQuery(String field, float[] target, int k) {
    super(field, target, (int) (k * OVERSAMPLE));
    this.target = target;
  }

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    this.searcher = indexSearcher;
    return super.rewrite(indexSearcher);
  }

  // XXX we catch IOException and then do nothing with it.
  @Override
  protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
    TopDocs topDocs = super.mergeLeafResults(perLeafResults);
    if (topDocs.scoreDocs.length == 0) {
      return topDocs;
    }

    // XXX we want to do the following:
    // * rescore query_float x doc_binary (kOversampled)
    // * rescore query_float x doc_float (k)
    // to do the former at each segment, we need to cast to CodecReader, getVectorReader(), cast
    // that to some other type, get a float/byte reader, duplicate.

    try {
      // XXX TaskExecutor taskExecutor = searcher.getTaskExecutor();
      TaskExecutor taskExecutor = TASK_EXECUTOR;
      List<LeafReaderContext> segments = this.searcher.getIndexReader().leaves();

      // topDocs.scoreDocs is the oversampled result of binary query x binary doc.
      // Rerank all of these results with float query x binary doc to improve fidelity,
      // Logic to re-score from the repository is cribbed from TopFieldCollector.
      // Logic to split into tasks is cribbed from AbstractKnnVectorQuery.
      scoreDocs(topDocs.scoreDocs, segments, this::approxScoreSegmentDocs, taskExecutor);

      // Truncate to the original K value, then rerank with float query x float doc.
      int originalK = (int) (getK() / OVERSAMPLE);
      topDocs.scoreDocs = Arrays.copyOf(topDocs.scoreDocs, originalK);
      scoreDocs(topDocs.scoreDocs, segments, this::fullScoreSegmentDocs, taskExecutor);
    } catch (IOException e) {
      /* do nothing and hope for the best */
    }

    return topDocs;
  }

  @FunctionalInterface
  private interface ScoreSegmentDocs {
    int score(LeafReaderContext ctx, ScoreDoc[] scoreDocs, int begin, int end) throws IOException;
  }

  private void scoreDocs(
      ScoreDoc[] scoreDocs,
      List<LeafReaderContext> segments,
      ScoreSegmentDocs segmentScorer,
      TaskExecutor taskExecutor)
      throws IOException {
    // Sort in doc order since we need to score everything in each segment together.
    Arrays.sort(scoreDocs, Comparator.comparingInt(sd -> sd.doc));
    List<Callable<Integer>> tasks = new ArrayList<>(segments.size());
    int i = 0;
    while (i < scoreDocs.length) {
      final int start = i;
      LeafReaderContext ctx = segments.get(ReaderUtil.subIndex(scoreDocs[i].doc, segments));
      final int maxDoc = ctx.docBase + ctx.reader().maxDoc();
      for (i = i + 1; i < scoreDocs.length; i++) {
        if (scoreDocs[i].doc >= maxDoc) {
          break;
        }
      }
      final int end = i;
      tasks.add(() -> segmentScorer.score(ctx, scoreDocs, start, end));
    }
    if (taskExecutor.invokeAll(tasks).stream().reduce(0, (a, b) -> a + b) == 0) {
      throw new IllegalStateException();
    }
    Arrays.sort(scoreDocs, (a, b) -> -Double.compare(a.score, b.score));
  }

  private static class ApproxFloatVectorScorer extends VectorScorer {
    private final float[] target;
    private final float[] doc;
    private final BinaryVectorValues vectorValues;

    ApproxFloatVectorScorer(
        VectorSimilarityFunction similarityFunction,
        float[] target,
        BinaryVectorValues vectorValues) {
      super(similarityFunction);
      this.target = target;
      this.doc = new float[vectorValues.dimension()];
      this.vectorValues = vectorValues;
    }

    @Override
    boolean advanceExact(int doc) throws IOException {
      return this.vectorValues.advance(doc) == doc;
    }

    @Override
    float score() throws IOException {
      long[] binDoc = this.vectorValues.vectorValue();
      for (int i = 0; i < this.doc.length; i++) {
        this.doc[i] = ((binDoc[i / 64] >> (i % 64)) & 0x1) == 0x1 ? 1.0f : -1.0f;
      }
      return this.similarity.compare(this.target, this.doc);
    }
  }

  private int approxScoreSegmentDocs(
      LeafReaderContext ctx, ScoreDoc[] scoreDocs, int begin, int end) throws IOException {
    // Try to get an underlying binary reader vector reader. Don't rescore if you can't do this.
    // NB: if you get here and the reader is not BQ you may have already mixed approximate and full
    // scored docs.
    if (!(ctx.reader() instanceof CodecReader)) {
      System.err.println("LeafReader:" + ctx.reader().getClass());
      return 0;
    }
    KnnVectorsReader vectorsReader = ((CodecReader) ctx.reader()).getVectorReader();
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader) {
      vectorsReader =
          ((PerFieldKnnVectorsFormat.FieldsReader) vectorsReader).getFieldReader(getField());
    }
    if (!(vectorsReader instanceof BinaryQuantizedVectorsReader)) {
      System.err.println("KnnVectorsReader:" + vectorsReader.getClass());
      return 0;
    }
    BinaryVectorValues vectorValues =
        ((BinaryQuantizedVectorsReader) vectorsReader).getBinaryVectorValues(getField());
    return scoreDocs(
        ctx,
        new ApproxFloatVectorScorer(
            getFieldInfo(ctx).getVectorSimilarityFunction(), this.target, vectorValues),
        scoreDocs,
        begin,
        end);
  }

  private int fullScoreSegmentDocs(LeafReaderContext ctx, ScoreDoc[] scoreDocs, int begin, int end)
      throws IOException {
    return scoreDocs(ctx, createVectorScorer(ctx, getFieldInfo(ctx)), scoreDocs, begin, end);
  }

  private FieldInfo getFieldInfo(LeafReaderContext ctx) {
    return ctx.reader().getFieldInfos().fieldInfo(getField());
  }

  private int scoreDocs(
      LeafReaderContext ctx, VectorScorer scorer, ScoreDoc[] scoreDocs, int begin, int end)
      throws IOException {
    for (int i = begin; i < end; i++) {
      if (!scorer.advanceExact(scoreDocs[i].doc - ctx.docBase)) {
        throw new IllegalArgumentException("Doc " + scoreDocs[i].doc + " doesn't have a vector");
      }
      scoreDocs[i].score = scorer.score();
    }
    return end - begin;
  }
}
