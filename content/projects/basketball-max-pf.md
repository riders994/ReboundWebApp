In fantasy football, there exists a concept of Max Points For (MPF) by looking at every week
and rearranging your lineup so that it achieves the most points possible. For the purposes of
fantasy sports, especially dynasty, it's a more reliable measure of talent in the long term
than wins alone. Many dynasty leagues use it as a metric to determine draft order to
discourage a "soft tank". I got Claude to write a python package to compute this for
basketball.

Since basketball can have category scoring available, this problem becomes more nebulous.
What is the "best" array of categories to score? While people are constantly trying to answer
this question from a drafting perspective (HashtagBasketball uses weighted Z-scores for their
grading), it's slightly easier answering this question post-draft. Looking at a particular
group of players over a period of time, it is not hard to determine which configuration
yields the most optimal output.

## The scores

**M1** — if you'd played the lineup that makes your team shine at what they're best at
against your opponents, how would you have fared. This will always be more than the original
number of categories won.

**M2** — the number of categories won if both players ran their perceived optimal lineup,
which can be lower than your actual score.

**M3** — a best guess at what it would look like if both teams were actively trying to win by
setting a lineup most likely to outscore their specific opponent.

There are a few other metrics worth calculating off of this which are mentioned in the repo
itself.
