package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;

/** I'm just here so the build doesn't fail. */
public class KnnFloatRescoreVectorQuery extends KnnFloatVectorQuery {
  private static final float OVERSAMPLE;

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

    try {
      // Logic to re-score from the repository is cribbed from TopFieldCollector.
      ScoreDoc scoreDocs[] = topDocs.scoreDocs.clone();
      Arrays.sort(scoreDocs, Comparator.comparingInt(sd -> sd.doc));
      List<LeafReaderContext> contexts = searcher.getIndexReader().leaves();
      TaskExecutor taskExecutor = searcher.getTaskExecutor();
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
      /*
      LeafReaderContext currentContext = null;
      VectorScorer currentScorer = null;
      // XXX we could select all docs that match a context, then use the task executor to re-score
      // them all. This would at least parallelize across segments.
      for (ScoreDoc scoreDoc : scoreDocs) {
        if (currentContext == null
            || scoreDoc.doc >= currentContext.docBase + currentContext.reader().maxDoc()) {
          int newContextIndex = ReaderUtil.subIndex(scoreDoc.doc, contexts);
          currentContext = contexts.get(newContextIndex);
          currentScorer =
              createVectorScorer(
                  currentContext, currentContext.reader().getFieldInfos().fieldInfo(getField()));
        }

        if (!currentScorer.advanceExact(scoreDoc.doc - currentContext.docBase)) {
          throw new IllegalArgumentException("Doc " + scoreDoc.doc + " doesn't have a vector");
        }
        scoreDoc.score = currentScorer.score();
      }
       */
      Arrays.sort(topDocs.scoreDocs, (a, b) -> -Double.compare(a.score, b.score));
      topDocs.scoreDocs = Arrays.copyOf(topDocs.scoreDocs, (int) (this.getK() / OVERSAMPLE));
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
