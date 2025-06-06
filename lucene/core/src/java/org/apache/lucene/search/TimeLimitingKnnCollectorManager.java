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
package org.apache.lucene.search;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/** A {@link KnnCollectorManager} that collects results with a timeout. */
public class TimeLimitingKnnCollectorManager implements KnnCollectorManager {
  private final KnnCollectorManager delegate;
  private final QueryTimeout queryTimeout;

  public TimeLimitingKnnCollectorManager(KnnCollectorManager delegate, QueryTimeout timeout) {
    this.delegate = delegate;
    this.queryTimeout = timeout;
  }

  /** Get the configured {@link QueryTimeout} for terminating graph and exact searches. */
  public QueryTimeout getQueryTimeout() {
    return queryTimeout;
  }

  @Override
  public KnnCollector newCollector(
      int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
      throws IOException {
    KnnCollector collector = delegate.newCollector(visitedLimit, searchStrategy, context);
    if (queryTimeout == null) {
      return collector;
    }
    return new TimeLimitingKnnCollector(collector);
  }

  class TimeLimitingKnnCollector extends KnnCollector.Decorator {
    public TimeLimitingKnnCollector(KnnCollector collector) {
      super(collector);
    }

    @Override
    public boolean earlyTerminated() {
      return queryTimeout.shouldExit() || super.earlyTerminated();
    }

    @Override
    public TopDocs topDocs() {
      TopDocs docs = super.topDocs();

      // Mark results as partial if timeout is met
      TotalHits.Relation relation =
          queryTimeout.shouldExit()
              ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO
              : docs.totalHits.relation();

      return new TopDocs(new TotalHits(docs.totalHits.value(), relation), docs.scoreDocs);
    }
  }
}
