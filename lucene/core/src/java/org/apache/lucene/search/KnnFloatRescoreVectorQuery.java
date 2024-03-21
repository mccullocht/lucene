package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;

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

    // XXX we re-score all the vectors in the oversample. we either do not want to score the
    // oversample OR we would like a lesser oversample variable to score.
    int originalK = (int) (getK() / OVERSAMPLE);
    try {
      // Logic to re-score from the repository is cribbed from TopFieldCollector.
      // Logic to split into tasks is cribbed from AbstractKnnVectorQuery.
      ScoreDoc scoreDocs[] = Arrays.copyOf(topDocs.scoreDocs, originalK);
      Arrays.sort(scoreDocs, Comparator.comparingInt(sd -> sd.doc));
      List<LeafReaderContext> contexts = searcher.getIndexReader().leaves();
      // XXX TaskExecutor taskExecutor = searcher.getTaskExecutor();
      TaskExecutor taskExecutor = TASK_EXECUTOR;
      List<Callable<Integer>> tasks = new ArrayList<>(contexts.size());
      int i = 0;
      while (i < scoreDocs.length) {
        final int start = i;
        LeafReaderContext ctx = contexts.get(ReaderUtil.subIndex(scoreDocs[i].doc, contexts));
        final int maxDoc = ctx.docBase + ctx.reader().maxDoc();
        for (i = i + 1; i < scoreDocs.length; i++) {
          if (scoreDocs[i].doc >= maxDoc) {
            break;
          }
        }
        final int end = i;
        tasks.add(() -> rescoreDocs(ctx, scoreDocs, start, end));
      }
      taskExecutor.invokeAll(tasks);
      Arrays.sort(scoreDocs, (a, b) -> -Double.compare(a.score, b.score));
      topDocs.scoreDocs = scoreDocs;
    } catch (IOException e) {
      /* do nothing and hope for the best */
    }

    return topDocs;
  }

  private int rescoreDocs(LeafReaderContext ctx, ScoreDoc[] scoreDocs, int begin, int end)
      throws IOException {
    VectorScorer scorer =
        createVectorScorer(ctx, ctx.reader().getFieldInfos().fieldInfo(getField()));
    for (int i = begin; i < end; i++) {
      if (!scorer.advanceExact(scoreDocs[i].doc - ctx.docBase)) {
        throw new IllegalArgumentException("Doc " + scoreDocs[i].doc + " doesn't have a vector");
      }
      scoreDocs[i].score = scorer.score();
    }
    return end - begin;
  }
}
