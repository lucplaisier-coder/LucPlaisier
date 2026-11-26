# Defence preparation (roadmap step 16)

The format is fixed by the guidelines (§4): **3 minutes per student**. That is up to **2 minutes** of
presentation, then **one short question** from the professor, with strictly enforced time limits.
Everything below uses only signed-off results. Slides: the "Capstone defence slides" deck (4 slides, speaker notes hold the script; downloads as PowerPoint or PDF).

---

## Part 1. The 2-minute talk (about 260 words, one slide per block)

Read it aloud with a timer. At a normal pace it takes 1:50 to 2:00. The **bold** words are the ones to
stress.

**[Slide 1: The question] (about 25 s)**
On Lending Club, investors choose which approved loans to fund. The platform already prices each loan
with a sub-grade and an interest rate. So my question is: **can machine learning find good and bad
loans *inside* the same price?** I compare four strategies. Each funds 80% of every grade: fund
everything, Lending Club's own ranking, a logistic regression, and gradient boosting.

**[Slide 2: How I kept it honest] (about 30 s)**
I used 611,000 finished 36-month loans. The models train on 2010 to 2013, using only the nine facts
investors saw at listing. **I froze the whole method before opening the test year.** That covered the
inputs, the settings and even the four possible verdicts. Then I opened 2014 once, and 2015 once as a
robustness check. The main metric is real **cash back per dollar invested**, after the platform's fee.

**[Slide 3: The result] (about 35 s)**
In 2014, boosting earned **8.01 cents per dollar**. The logit earned 7.95, Lending Club's ranking 7.68
and funding everything 7.74. Both pre-registered comparisons are above zero in the bootstrap, so the
verdict is **"AI adds economic value"**. But be precise about size. Beating the platform is worth
about **$33 per $10,000**. Boosting over the logit adds only about **$5**. 2015 gives the same verdict
with smaller gains.

**[Slide 4: What it means] (about 30 s)**
The models don't replace the platform's price. They **refine** it with income, loan size, debt and
FICO. Lending Club's own ranking inside a grade actually lost money: its highest-priced loans paid for
their risk. The limits are a simple money measure, rising defaults after 2013, and a bootstrap that
may understate uncertainty. **The whole result reruns from the raw file with one command.**

---

## Part 2. Likely questions, with short answers (15-30 seconds each)

You get **one** question. Each answer starts with the main point in one sentence, then gives at most
two supporting facts.

### About the design

**1. Why select inside each grade instead of across all loans?**
Otherwise a model could "win" just by avoiding risky grades and taking lower interest. Keeping 80% of
every grade fixes the risk mix. Any gain must then come from picking better loans at the same price.
That is exactly the information the platform's price might miss.

**2. Why not a random train/test split?**
Loans issued later can't help predict earlier loans, and a random split would mix them. So I split by
year: train 2010-12, tune on 2013, test 2014, robustness 2015. That is how an investor would really
use it.

**3. Why only nine inputs? Weren't there 151 columns?**
Most columns are recorded *after* the loan was issued, such as payments and recoveries. Using them is
leakage. I kept only fields investors saw at listing, checked against Lending Club's "browseNotes"
list, and froze them before fitting any model.

**4. Why did you drop verification status?**
The data does not say *when* the verification finished. It may have finished after investors
committed, so I couldn't prove it was known at decision time. I dropped it as a timing precaution,
before any model was fitted.

**5. Why HistGB and not a neural network or XGBoost?**
HistGB is the course's default tree ensemble, and it handles missing values natively. My question is
whether *any* flexible model adds value beyond a logit. With nine mostly monotone inputs, there is
little left for a bigger model to find. Boosting already gained only 0.05 cents.

**6. How did you tune it? Did you avoid overfitting?**
I used a small, fixed grid tuned on 2013 only. The rule was: best AUC, and within 0.002 the simpler
model wins. The smallest model was also the best: depth 3, 200 trees. Larger ones did *worse* on
2013. Early stopping was off, because 2013 is later reused for the final fit.

**7. What if HGB had lost?**
I would have reported it. The four verdicts, including "simpler model preferable" and "platform
pricing hard to beat", were fixed *before* the test. The freeze record has a hash and a timestamp.
2014 was opened exactly once.

### About the results

**8. 0.05 cents is tiny. Why does it matter?**
It is small, and I say so. It is about $5 per $10,000 over the loans' life. The important result is
the 0.33 cents (about $33 per $10,000) over Lending Club's own ranking. Practically, the logit gets
most of the gain and is easier to explain.

**9. Why is your interval so close to zero (+0.009)?**
The bootstrap resamples single loans, so it ignores shocks shared by loans issued in the same month. I
state this as a limitation. With a clustered bootstrap, the HGB-minus-logit interval might include
zero. The gain over the platform, from 0.26 to 0.40 cents, is much further from zero.

**10. Why does Lending Club's ranking lose money?**
Inside a grade, it skips the highest-rate sub-grades first. Those loans default more, but their extra
interest more than paid for it in 2013 and 2014. So the platform's fine pricing inside a grade does
not mark *worse investments*. That is why a model can add value there.

**11. Why not a Sharpe ratio or annual return?**
The data says how much each borrower paid, but not *when*, so I can't compute an honest annual return
or IRR. Each strategy is a buy-and-hold loan portfolio with no market prices, so Sharpe ratio and
drawdown don't apply. I report cash back per dollar after the 1% fee, plus a 0% fee check.

**12. Where does the 1% fee come from?**
It is Lending Club's investor service fee on payments received. It is an assumption, so I also report
0%. Every conclusion is unchanged.

**13. Your models are badly calibrated in 2015. Isn't that a problem?**
Defaults rose from 12.4% in training to 14.9% in 2015, so both models predict too little risk. But
selection uses only the *ranking* of loans, and the ranking still works: AUC 0.68. A live system
would need recalibration.

**14. What happened in the riskiest grades in 2015?**
In grades D-G, the platform rule did best in 2015, so the model's edge there did not hold. These are
point estimates without intervals, so small gaps shouldn't be over-read. Still, it is an honest
weakness, and I report it.

### About interpretation and deployment

**15. What drives the models?**
Mostly the platform's own price. Shuffling sub-grade and rate together costs 0.095 AUC. On top of
that price, income matters most. At the same price, higher income means about 24% lower odds of
charge-off per standard deviation. Loan size, debt-to-income and FICO follow.

**16. Why shuffle sub-grade and rate together?**
They carry almost the same information. Shuffled alone, the sub-grade looks unimportant (0.005),
because the rate covers for it. Shuffled together, the true reliance on the platform's price shows
(0.095). I fixed this choice before the test.

**17. Could an investor really use this?**
Partly. Popular loans were often funded first by faster investors, and Lending Club sold some loans
whole to institutions. It also closed its retail notes platform in 2020. So this is evidence about
information in the data, not a ready trading system.

**18. Is it fair or legal?**
The model uses no protected characteristics, and the investor doesn't decide who gets credit. Still,
inputs like income can correlate with protected groups (Fuster et al., 2022). A lender using it would
face fair-lending and explainability rules. That is one more reason to prefer the transparent logit.

**19. How do I know your numbers are real?**
One command, `python -m src.reproduce_all`, reruns everything from the raw file in about 8 minutes.
It checks that every table and figure is identical. I ran it myself in Colab, and it printed
`REPRODUCED`.

**20. What did AI tools do in this project?**
ChatGPT and Claude helped review the design, and Claude wrote and ran the code under my direction and
drafted the text. Every decision was mine and is on record with a date:
the input freeze and the method freeze before the test, and a sign-off after each step. I can explain every step.

---

## Part 3. Numbers to know by heart

| What | Number |
|---|---|
| Loans in the study | 611,816 (36-month, 2010-2015) |
| Test year 2014 | 162,570 loans, 13.7% charged off |
| Net cash per $ in 2014 | HGB 8.01 c · Logit 7.95 c · Fund-all 7.74 c · Platform 7.68 c |
| HGB − Logit | +0.054 c [+0.009, +0.100] (about $5 per $10,000) |
| HGB − Platform | +0.332 c [+0.259, +0.404] (about $33 per $10,000) |
| AUC 2014 | HGB 0.672 · Logit 0.670 (sub-grade alone: 0.649 on 2013) |
| 2015 | HGB 5.48 c · Logit 5.44 c · Platform 5.34 c · Fund-all 5.30 c; same verdict |
| Strongest borrower fact | Income: odds ratio 0.76 per SD |
| Pricing block importance | 0.095 AUC drop |
| Frozen settings | Logit C = 0.01; HGB depth 3, 200 trees, learning rate 0.05 |
