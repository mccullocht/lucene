package org.apache.lucene.search;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;

/** I'm just here so the build doesn't fail. */
public class KnnFloatRescoreVectorQuery extends KnnFloatVectorQuery {
  // NB: the superclass has this as a private member.
  private final float[] target;
  // NB: captured to access segment state to re-score.
  private IndexSearcher searcher;

  public KnnFloatRescoreVectorQuery(String field, float[] target, int k) {
    super(field, target, k);
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
      LeafReaderContext currentContext = null;
      VectorScorer currentScorer = null;
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
      Arrays.sort(topDocs.scoreDocs, (a, b) -> -Double.compare(a.score, b.score));
    } catch (IOException e) {
      /* do nothing and hope for the best */
    }

    return topDocs;
  }
}
