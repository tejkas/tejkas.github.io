---
title: 'Where the p90 went: profiling a slow query planner'
description: 'The median was fine, so nobody looked. A flame graph found 40% of tail latency in a function that was supposed to be a cache hit.'
pubDate: 2026-07-22
tags: ['databases', 'performance']
draft: true
---

Our planner had a median latency of 3ms and a p90 of 210ms. Nobody had looked
at it closely because the median looked healthy, which is exactly the trap.

## Reading the shape, not the number

A 70× gap between median and p90 usually means two populations, not one slow
path getting slower. Plotting the latency histogram on a log axis showed the
second hump clearly, sitting around 200ms.

The question then isn't "why is the planner slow" but "what distinguishes the
requests in the second hump".

## The answer was cardinality estimation

Statistics lookups were supposed to hit an in-memory cache. They did — for
tables the planner had seen since the last stats refresh. Every refresh
invalidated the whole cache, so the first query touching each table after a
refresh paid a full catalog read.

Refreshes ran every 5 minutes. With roughly 40 hot tables and a few thousand
queries per interval, about 1% of queries paid the cost, which lands almost
exactly where the second hump was.

## The fix, and the fix after that

Refreshing entries individually instead of dropping the whole map took p90 to
34ms. Prewarming the cache in the background after a refresh took it to 6ms.

The thing I'd want to remember: the bug was visible in the histogram from the
start and invisible in every average we had on a dashboard. We'd been
monitoring the metric that couldn't show it.
