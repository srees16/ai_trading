# Chapter 19: Microstructural Features


CHAPTER 19
Microstructural Features
19.1
MOTIVATION
Market microstructure studies “the process and outcomes of exchanging assets under
explicit trading rules” (O’Hara [1995]). Microstructural datasets include primary
information about the auctioning process, like order cancellations, double auction
book, queues, partial fills, aggressor side, corrections, replacements, etc. The main
source is Financial Information eXchange (FIX) messages, which can be purchased
from exchanges. The level of detail contained in FIX messages provides researchers
with the ability to understand how market participants conceal and reveal their inten-
tions. That makes microstructural data one of the most important ingredients for
building predictive ML features.
19.2
REVIEW OF THE LITERATURE
The depth and complexity of market microstructure theories has evolved over time,
as a function of the amount and variety of the data available. The first generation of
models used solely price information. The two foundational results from those early
days are trade classification models (like the tick rule) and the Roll [1984] model. The
second generation of models came after volume datasets started to become available,
and researchers shifted their attention to study the impact that volume has on prices.
Two examples for this generation of models are Kyle [1985] and Amihud [2002].
The third generation of models came after 1996, when Maureen O’Hara, David
Easley, and others published their “probability of informed trading” (PIN) theory
(Easley et al. [1996]). This constituted a major breakthrough, because PIN explained
the bid-ask spread as the consequence of a sequential strategic decision between liq-
uidity providers (market makers) and position takers (informed traders). Essentially,
it illustrated that market makers were sellers of the option to be adversely selected by
281


282
MICROSTRUCTURAL FEATURES
informed traders, and the bid-ask spread is the premium they charge for that option.
Easley et al. [2012a, 2012b] explain how to estimate VPIN, a high-frequency estimate
of PIN under volume-based sampling.
These are the main theoretical frameworks used by the microstructural literature.
O’Hara [1995] and Hasbrouck [2007] offer a good compendium of low-frequency
microstructural models. Easley et al. [2013] present a modern treatment of high-
frequency microstructural models.
19.3
FIRST GENERATION: PRICE SEQUENCES
The first generation of microstructural models concerned themselves with estimating
the bid-ask spread and volatility as proxies for illiquidity. They did so with limited
data and without imposing a strategic or sequential structure to the trading process.
19.3.1
The Tick Rule
In a double auction book, quotes are placed for selling a security at various price levels
(offers) or for buying a security at various price levels (bids). Offer prices always
exceed bid prices, because otherwise there would be an instant match. A trade occurs
whenever a buyer matches an offer, or a seller matches a bid. Every trade has a buyer
and a seller, but only one side initiates the trade.
The tick rule is an algorithm used to determine a trade’s aggressor side. A buy-
initiated trade is labeled “1”, and a sell-initiated trade is labeled “-1”, according to
this logic:
bt =
⎧
⎪
⎨
⎪⎩
1
if Δpt > 0
−1
if Δpt < 0
bt−1
if Δpt = 0
where pt is the price of the trade indexed by t = 1, …, T, and b0 is arbitrarily set to
1. A number of studies have determined that the tick rule achieves high classifica-
tion accuracy, despite its relative simplicity (Aitken and Frino [1996]). Competing
classification methods include Lee and Ready [1991] and Easley et al. [2016].
Transformations of the {bt} series can result in informative features. Such trans-
formations include: (1) Kalman Filters on its future expected value, Et[bt+1]; (2)
structural breaks on such predictions (Chapter 17), (3) entropy of the {bt} sequence
(Chapter 18); (4) t-values from Wald-Wolfowitz’s tests of runs on {bt}; (5) fractional
differentiation of the cumulative {bt} series, ∑t
i=1 bi (Chapter 5); etc.
19.3.2
The Roll Model
Roll [1984] was one of the first models to propose an explanation for the effective
bid-ask spread at which a security trades. This is useful in that bid-ask spreads are a
function of liquidity, hence Roll’s model can be seen as an early attempt to measure


FIRST GENERATION: PRICE SEQUENCES
283
the liquidity of a security. Consider a mid-price series {mt}, where prices follow a
Random Walk with no drift,
mt = mt−1 + ut
hence price changes Δmt = mt −mt−1 are independently and identically drawn from
a Normal distribution
Δmt∼N [0, 𝜎2
u
]
These assumptions are, of course, against all empirical observations, which sug-
gest that financial time series have a drift, they are heteroscedastic, exhibit serial
dependency, and their returns distribution is non-Normal. But with a proper sampling
procedure, as we saw in Chapter 2, these assumptions may not be too unrealistic. The
observed prices, {pt}, are the result of sequential trading against the bid-ask spread:
pt = mt + btc
where c is half the bid-ask spread, and bt ∈{−1, 1} is the aggressor side. The Roll
model assumes that buys and sells are equally likely, P[bt = 1] = P[bt = −1] = 1
2,
serially independent, E[btbt−1] = 0, and independent from the noise, E[btut] = 0.
Given these assumptions, Roll derives the values of c and 𝜎2
u as follows:
𝜎2 [Δpt
] = E
[(Δpt
)2]
−(E [(Δpt
)])2 = 2c2 + 𝜎2
u
𝜎[Δpt, Δpt−1
] = −c2
resulting in c =
√
max{0, −𝜎[Δpt, Δpt−1]} and 𝜎2
u = 𝜎2[Δpt] + 2𝜎[Δpt, Δpt−1]. In
conclusion, the bid-ask spread is a function of the serial covariance of price changes,
and the true (unobserved) price’s noise, excluding microstructural noise, is a function
of the observed noise and the serial covariance of price changes.
The reader may question the need for Roll’s model nowadays, when datasets
include bid-ask prices at multiple book levels. One reason the Roll model is still
in use, despite its limitations, is that it offers a relatively direct way to determine the
effective bid-ask spread of securities that are either rarely traded, or where the pub-
lished quotes are not representative of the levels at which market makers’ are willing
to provide liquidity (e.g., corporate, municipal, and agency bonds). Using Roll’s esti-
mates, we can derive informative features regarding the market’s liquidity conditions.
19.3.3
High-Low Volatility Estimator
Beckers [1983] shows that volatility estimators based on high-low prices are more
accurate than the standard estimators of volatility based on closing prices. Parkinson


284
MICROSTRUCTURAL FEATURES
[1980] derives that, for continuously observed prices following a geometric Brownian
motion,
E
[
1
T
T
∑
t=1
(
log
[Ht
Lt
])2]
= k1𝜎2
HL
E
[
1
T
T
∑
t=1
(
log
[Ht
Lt
])]
= k2𝜎HL
where k1 = 4log[2], k2 =
√
8
𝜋, Ht is the high price for bar t, and Lt is the low price
for bar t. Then the volatility feature 𝜎HL can be robustly estimated based on observed
high-low prices.
19.3.4
Corwin and Schultz
Building on the work of Beckers [1983], Corwin and Schultz [2012] introduce a bid-
ask spread estimator from high and low prices. The estimator is based on two prin-
ciples: First, high prices are almost always matched against the offer, and low prices
are almost always matched against the bid. The ratio of high-to-low prices reflects
fundamental volatility as well as the bid-ask spread. Second, the component of the
high-to-low price ratio that is due to volatility increases proportionately with the time
elapsed between two observations.
Corwin and Schultz show that the spread, as a factor of price, can be estimated as
St = 2 (e𝛼t −1)
1 + e𝛼t
where
𝛼t =
√
2𝛽t −
√
𝛽t
3 −2
√
2
−
√
𝛾t
3 −2
√
2
𝛽t = E
[ 1
∑
j=0
[
log
(Ht−j
Lt−j
)]2]
𝛾t =
[
log
(Ht−1,t
Lt−1,t
)]2
and Ht−1,t is the high price over 2 bars (t −1 and t), whereas Lt−1,t is the low price over
2 bars (t −1 and t). Because 𝛼t < 0 ⇒St < 0, the authors recommend setting negative


FIRST GENERATION: PRICE SEQUENCES
285
alphas to 0 (see Corwin and Schultz [2012], p. 727). Snippet 19.1 implements this
algorithm. The corwinSchultz function receives two arguments, a series dataframe
with columns (High,Low), and an integer value sl that defines the sample length used
to estimate 𝛽t.
SNIPPET 19.1
IMPLEMENTATION OF THE CORWIN-SCHULTZ
ALGORITHM
def getBeta(series,sl):
hl=series[['High','Low']].values
hl=np.log(hl[:,0]/hl[:,1])**2
hl=pd.Series(hl,index=series.index)
beta=pd.stats.moments.rolling_sum(hl,window=2)
beta=pd.stats.moments.rolling_mean(beta,window=sl)
return beta.dropna()
#———————————————————————————————————————-
def getGamma(series):
h2=pd.stats.moments.rolling_max(series['High'],window=2)
l2=pd.stats.moments.rolling_min(series['Low'],window=2)
gamma=np.log(h2.values/l2.values)**2
gamma=pd.Series(gamma,index=h2.index)
return gamma.dropna()
#———————————————————————————————————————-
def getAlpha(beta,gamma):
den=3–2*2**.5
alpha=(2**.5–1)*(beta**.5)/den
alpha-=(gamma/den)**.5
alpha[alpha<0]=0 # set negative alphas to 0 (see p.727 of paper)
return alpha.dropna()
#———————————————————————————————————————-
def corwinSchultz(series,sl=1):
# Note: S<0 iif alpha<0
beta=getBeta(series,sl)
gamma=getGamma(series)
alpha=getAlpha(beta,gamma)
spread=2*(np.exp(alpha)-1)/(1+np.exp(alpha))
startTime=pd.Series(series.index[0:spread.shape[0]],index=spread.index)
spread=pd.concat([spread,startTime],axis=1)
spread.columns=['Spread','Start_Time'] # 1st loc used to compute beta
return spread
Note that volatility does not appear in the final Corwin-Schultz equations. The
reason is that volatility has been replaced by its high/low estimator. As a byproduct
of this model, we can derive the Becker-Parkinson volatility as shown in Snippet 19.2.


286
MICROSTRUCTURAL FEATURES
SNIPPET 19.2
ESTIMATING VOLATILITY FOR HIGH-LOW PRICES
def getSigma(beta,gamma):
k2=(8/np.pi)**.5
den=3–2*2**.5
sigma=(2**-.5–1)*beta**.5/(k2*den)
sigma+=(gamma/(k2**2*den))**.5
sigma[sigma<0]=0
return sigma
This procedure is particularly helpful in the corporate bond market, where there
is no centralized order book, and trades occur through bids wanted in competition
(BWIC). The resulting feature, bid-ask spread S, can be estimated recursively over a
rolling window, and values can be smoothed using a Kalman filter.
19.4
SECOND GENERATION: STRATEGIC TRADE MODELS
Second generation microstructural models focus on understanding and measuring
illiquidity. Illiquidity is an important informative feature in financial ML models,
because it is a risk that has an associated premium. These models have a stronger
theoretical foundation than first-generation models, in that they explain trading as
the strategic interaction between informed and uninformed traders. In doing so, they
pay attention to signed volume and order flow imbalance.
Most of these features are estimated through regressions. In practice, I have
observed that the t-values associated with these microstructural estimates are more
informative than the (mean) estimates themselves. Although the literature does not
mention this observation, there is a good argument for preferring features based on
t-values over features based on mean values: t-values are re-scaled by the standard
deviation of the estimation error, which incorporates another dimension of informa-
tion absent in mean estimates.
19.4.1
Kyle’s Lambda
Kyle [1985] introduced the following strategic trade model. Consider a risky asset
with terminal value v ∼N[p0, Σ0], as well as two traders:
r A noise trader who trades a quantity u = N[0, 𝜎2
u], independent of v.
r An informed trader who knows v and demands a quantity x, through a market
order.
The market maker observes the total order flow y = x + u, and sets a price p accord-
ingly. In this model, market makers cannot distinguish between orders from noise


SECOND GENERATION: STRATEGIC TRADE MODELS
287
traders and informed traders. They adjust prices as a function of the order flow imbal-
ance, as that may indicate the presence of an informed trader. Hence, there is a positive
relationship between price change and order flow imbalance, which is called market
impact.
The informed trader conjectures that the market maker has a linear price adjust-
ment function, p = 𝜆y + 𝜇, where 𝜆is an inverse measure of liquidity. The informed
trader’s profits are 𝜋= (v −p)x, which are maximized at x = v−𝜇
2𝜆, with second order
condition 𝜆> 0.
Conversely, the market maker conjectures that the informed trader’s demand is a
linear function of v: x = 𝛼+ 𝛽v, which implies 𝛼= −𝜇
2𝜆and 𝛽=
1
2𝜆. Note that lower
liquidity means higher 𝜆, which means lower demand from the informed trader.
Kyle argues that the market maker must find an equilibrium between profit max-
imization and market efficiency, and that under the above linear functions, the only
possible solution occurs when
𝜇= p0
𝛼= p0
√
𝜎2
u
Σ0
𝜆= 1
2
√
Σ0
𝜎2
u
𝛽=
√
𝜎2
u
Σ0
Finally, the informed trader’s expected profit can be rewritten as
E [𝜋] =
(v −p0
)2
2
√
𝜎2
u
Σ0
= 1
4𝜆
(v −p0
)2
The implication is that the informed trader has three sources of profit:
r The security’s mispricing.
r The variance of the noise trader’s net order flow. The higher the noise, the easier
the informed trader can conceal his intentions.
r The reciprocal of the terminal security’s variance. The lower the volatility, the
easier to monetize the mispricing.


288
MICROSTRUCTURAL FEATURES
2.00
1e–3
1.75
1.50
1.25
1.00
0.75
0.50
0.25
0.00
0
2000
4000
6000
8000
10000
12000
14000
16000
FIGURE 19.1
Kyle’s Lambdas Computed on E-mini S&P 500 Futures
In Kyle’s model, the variable 𝜆captures price impact. Illiquidity increases with
uncertainty about v and decreases with the amount of noise. As a feature, it can be
estimated by fitting the regression
Δpt = 𝜆(btVt
) + 𝜀t
where {pt} is the time series of prices, {bt} is the time series of aggressor flags, {Vt} is
the time series of traded volumes, and hence {btVt} is the time series of signed volume
or net order flow. Figure 19.1 plots the histogram of Kyle’s lambdas estimated on the
E-mini S&P 500 futures series.
19.4.2
Amihud’s Lambda
Amihud [2002] studies the positive relationship between absolute returns and illiq-
uidity. In particular, he computes the daily price response associated with one dollar
of trading volume, and argues its value is a proxy of price impact. One possible imple-
mentation of this idea is
|||Δlog [̃p𝜏
]||| = 𝜆
∑
t∈B𝜏
(ptVt
) + 𝜀𝜏
where B𝜏is the set of trades included in bar 𝜏, ̃p𝜏is the closing price of bar 𝜏, and
ptVt is the dollar volume involved in trade t ∈B𝜏. Despite its apparent simplicity,
Hasbrouck [2009] found that daily Amihud’s lambda estimates exhibit a high rank


SECOND GENERATION: STRATEGIC TRADE MODELS
289
1.00
1e–9
0.6
0.8
0.4
0.2
0.0
0
5000
10000
15000
20000
FIGURE 19.2
Amihud’s lambdas estimated on E-mini S&P 500 futures
correlation to intraday estimates of effective spread. Figure 19.2 plots the histogram
of Amihud’s lambdas estimated on the E-mini S&P 500 futures series.
19.4.3
Hasbrouck’s Lambda
Hasbrouck [2009] follows up on Kyle’s and Amihud’s ideas, and applies them to
estimating the price impact coefficient based on trade-and-quote (TAQ) data. He uses
a Gibbs sampler to produce a Bayesian estimation of the regression specification
log [̃pi,𝜏
] −log [̃pi,𝜏−1
] = 𝜆i
∑
t∈Bi,𝜏
(
bi,t
√
pi,tVi,t
)
+ 𝜀i,𝜏
where Bi,𝜏is the set of trades included in bar 𝜏for security i, with i = 1, …, I, ̃pi,𝜏is
the closing price of bar 𝜏for security i, bi,t ∈{−1, 1} indicates whether trade t ∈Bi,𝜏
was buy-initiated or sell-initiated; and pi,tVi,t is the dollar volume involved in trade
t ∈Bi,𝜏. We can then estimate 𝜆i for every security i, and use it as a feature that
approximates the effective cost of trading (market impact).
Consistent with most of the literature, Hasbrouck recommends 5-minute time-bars
for sampling ticks. However, for the reasons discussed in Chapter 2, better results can
be achieved through stochastic sampling methods that are synchronized with market
activity. Figure 19.3 plots the histogram of Hasbrouck’s lambdas estimated on the
E-mini S&P 500 futures series.


290
MICROSTRUCTURAL FEATURES
1e–7
4
3
2
1
0
0
2500
5000
7500
10000
12500
15000
17500
FIGURE 19.3
Hasbrouck’s lambdas estimated on E-mini S&P 500 futures
19.5
THIRD GENERATION: SEQUENTIAL TRADE MODELS
As we have seen in the previous section, strategic trade models feature a single
informed trader who can trade at multiple times. In this section we will discuss
an alternative kind of model, where randomly selected traders arrive at the market
sequentially and independently.
Since their appearance, sequential trade models have become very popular among
market makers. One reason is, they incorporate the sources of uncertainty faced by
liquidity providers, namely the probability that an informational event has taken
place, the probability that such event is negative, the arrival rate of noise traders,
and the arrival rate of informed traders. With those variables, market makers must
update quotes dynamically, and manage their inventories.
19.5.1
Probability of Information-based Trading
Easley et al. [1996] use trade data to determine the probability of information-based
trading (PIN) of individual securities. This microstructure model views trading as
a game between market makers and position takers that is repeated over multiple
trading periods.
Denote a security’s price as S, with present value S0. However, once a certain
amount of new information has been incorporated into the price, S will be either SB
(bad news) or SG (good news). There is a probability 𝛼that new information will
arrive within the timeframe of the analysis, a probability 𝛿that the news will be bad,


THIRD GENERATION: SEQUENTIAL TRADE MODELS
291
and a probability (1 −𝛿) that the news will be good. These authors prove that the
expected value of the security’s price can then be computed at time t as
E [St
] = (1 −𝛼t
) S0 + 𝛼t
[𝛿tSB + (1 −𝛿t
) SG
]
Following a Poisson distribution, informed traders arrive at a rate 𝜇, and unin-
formed traders arrive at a rate 𝜀. Then, in order to avoid losses from informed traders,
market makers reach breakeven at a bid level Bt,
E [Bt
] = E [St
] −
𝜇𝛼t𝛿t
𝜀+ 𝜇𝛼t𝛿t
(E [St
] −SB
)
and the breakeven ask level At at time t must be,
E [At
] = E [St
] +
𝜇𝛼t
(1 −𝛿t
)
𝜀+ 𝜇𝛼t
(1 −𝛿t
)
(SG −E [St
])
It follows that the breakeven bid-ask spread is determined as
E [At −Bt
] =
𝜇𝛼t
(1 −𝛿t
)
𝜀+ 𝜇𝛼t
(1 −𝛿t
)
(SG −E [St
]) +
𝜇𝛼t𝛿t
𝜀+ 𝜇𝛼t𝛿t
(E [St
] −SB
)
For the standard case when 𝛿t = 1
2, we obtain
𝛿t = 1
2 ⇒E [At −Bt
] =
𝛼t𝜇
𝛼t𝜇+ 2𝜀
(SG −SB
)
This equation tells us that the critical factor that determines the price range at
which market makers provide liquidity is
PINt =
𝛼t𝜇
𝛼t𝜇+ 2𝜀
The subscript t indicates that the probabilities 𝛼and 𝛿are estimated at that point
in time. The authors apply a Bayesian updating process to incorporate information
after each trade arrives to the market.
In order to determine the value PINt, we must estimate four non-observable param-
eters, namely {𝛼, 𝛿, 𝜇, 𝜀}. A maximum-likelihood approach is to fit a mixture of three
Poisson distributions,
P[VB, VS] = (1 −𝛼)P[VB, 𝜀]P[VS, 𝜀]
+ 𝛼(𝛿P[VB, 𝜀]P[VS, 𝜇+ 𝜀] + (1 −𝛿)P[VB, 𝜇+ 𝜀]P[VS, 𝜀])


292
MICROSTRUCTURAL FEATURES
where VB is the volume traded against the ask (buy-initiated trades), and VS is the
volume traded against the bid (sell-initiated trades).
19.5.2
Volume-Synchronized Probability of Informed Trading
Easley et al. [2008] proved that
E [VB −VS] = (1 −𝛼) (𝜀−𝜀) + 𝛼(1 −𝛿) (𝜀−(𝜇+ 𝜀)) + 𝛼𝛿(𝜇+ 𝜀−𝜀)
= 𝛼𝜇(1 −2𝛿)
and in particular, for a sufficiently large 𝜇,
E[|VB −VS|] ≈𝛼𝜇
Easley et al. [2011] proposed a high-frequency estimate of PIN, which they named
volume-synchronized probability of informed trading (VPIN). This procedure adopts
a volume clock, which synchronizes the data sampling with market activity, as cap-
tured by volume (see Chapter 2). We can then estimate
1
n
n
∑
𝜏=1
|||VB
𝜏−VS
𝜏
||| ≈𝛼𝜇
where VB
𝜏is the sum of volumes from buy-initiated trades within volume bar 𝜏, VS
𝜏is
the sum of volumes from sell-initiated trades within volume bar 𝜏, and n is the number
of bars used to produce this estimate. Because all volume bars are of the same size,
V, we know that by construction
1
n
n
∑
𝜏=1
(VB
𝜏+ VS
𝜏
) = V = 𝛼𝜇+ 2𝜀
Hence, PIN can be estimated in high-frequency as
VPIN𝜏=
∑n
𝜏=1 ||VB
𝜏−VS
𝜏||
∑n
𝜏=1
(VB
𝜏+ VS
𝜏
) =
∑n
𝜏=1 ||VB
𝜏−VS
𝜏||
nV
For additional details and case studies of VPIN, see Easley et al. [2013]. Using
linear regressions, Andersen and Bondarenko [2013] concluded that VPIN is not a
good predictor of volatility. However, a number of studies have found that VPIN
indeed has predictive power: Abad and Yague [2012], Bethel et al. [2012], Cheung
et al. [2015], Kim et al. [2014], Song et al. [2014], Van Ness et al. [2017], and Wei
et al. [2013], to cite a few. In any case, linear regression is a technique that was already
known to 18th-century mathematicians (Stigler [1981]), and economists should not
be surprised when it fails to recognize complex non-linear patterns in 21st-century
financial markets.


ADDITIONAL FEATURES FROM MICROSTRUCTURAL DATASETS
293
19.6
ADDITIONAL FEATURES FROM MICROSTRUCTURAL
DATASETS
The features we have studied in Sections 19.3 to 19.5 were suggested by mar-
ket microstructure theory. In addition, we should consider alternative features that,
although not suggested by the theory, we suspect carry important information about
the way market participants operate, and their future intentions. In doing so, we will
harness the power of ML algorithms, which can learn how to use these features with-
out being specifically directed by theory.
19.6.1
Distibution of Order Sizes
Easley et al. [2016] study the frequency of trades per trade size, and find that
trades with round sizes are abnormally frequent. For example, the frequency rates
quickly decay as a function of trade size, with the exception of round trade sizes
{5, 10, 20, 25, 50, 100, 200, …}. These authors attribute this phenomenon to so-
called “mouse” or “GUI” traders, that is, human traders who send orders by clicking
buttons on a GUI (Graphical User Interface). In the case of the E-mini S&P 500, for
example, size 10 is 2.9 times more frequent than size 9; size 50 is 10.9 times more
likely than size 49; size 100 is 16.8 times more frequent than size 99; size 200 is 27.2
times more likely than size 199; size 250 is 32.5 times more frequent than size 249;
size 500 is 57.1 times more frequent than size 499. Such patterns are not typical of
“silicon traders,” who usually are programmed to randomize trades to disguise their
footprint in markets.
A useful feature may be to determine the normal frequency of round-sized trades,
and monitor deviations from that expected value. The ML algorithm could, for exam-
ple, determine if a larger-than-usual proportion of round-sized trades is associated
with trends, as human traders tend to bet with a fundamental view, belief, or convic-
tion. Conversely, a lower-than-usual proportion of round-sized trades may increase
the likelihood that prices will move sideways, as silicon traders do not typically hold
long-term views.
19.6.2
Cancellation Rates, Limit Orders, Market Orders
Eisler et al. [2012] study the impact of market orders, limit orders, and quote can-
cellations. These authors find that small stocks respond differently than large stocks
to these events. They conclude that measuring these magnitudes is relevant to model
the dynamics of the bid-ask spread.
Easley et al. [2012] also argue that large quote cancellation rates may be indicative
of low liquidity, as participants are publishing quotes that do not intend to get filled.
They discuss four categories of predatory algorithms:
r Quote stuffers: They engage in “latency arbitrage.” Their strategy involves
overwhelming an exchange with messages, with the sole intention of slowing
down competing algorithms, which are forced to parse messages that only the
originators know can be ignored.


294
MICROSTRUCTURAL FEATURES
r Quote danglers: This strategy sends quotes that force a squeezed trader to
chase a price against her interests. O’Hara [2011] presents evidence of their
disruptive activities.
r Liquidity squeezers: When a distressed large investor is forced to unwind her
position, predatory algorithms trade in the same direction, draining as much
liquidity as possible. As a result, prices overshoot and they make a profit (Carlin
et al. [2007]).
r Pack hunters: Predators hunting independently become aware of one another’s
activities, and form a pack in order to maximize the chances of triggering a cas-
cading effect (Donefer [2010], Fabozzi et al. [2011], Jarrow and Protter [2011]).
NANEX [2011] shows what appears to be pack hunters forcing a stop loss.
Although their individual actions are too small to raise the regulator’s suspi-
cion, their collective action may be market-manipulative. When that is the case,
it is very hard to prove their collusion, since they coordinate in a decentralized,
spontaneous manner.
These predatory algorithms utilize quote cancellations and various order types in
an attempt to adversely select market makers. They leave different signatures in the
trading record, and measuring the rates of quote cancellation, limit orders, and market
orders can be the basis for useful features, informative of their intentions.
19.6.3
Time-Weighted Average Price Execution Algorithms
Easley et al. [2012] demonstrate how to recognize the presence of execution algo-
rithms that target a particular time-weighted average price (TWAP). A TWAP algo-
rithm is an algorithm that slices a large order into small ones, which are submitted
at regular time intervals, in an attempt to achieve a pre-defined time-weighted aver-
age price. These authors take a sample of E-mini S&P 500 futures trades between
November 7, 2010, and November 7, 2011. They divide the day into 24 hours, and
for every hour, they add the volume traded at each second, irrespective of the minute.
Then they plot these aggregate volumes as a surface where the x-axis is assigned
to volume per second, the y-axis is assigned to hour of the day, and the z-axis is
assigned to the aggregate volume. This analysis allows us to see the distribution of
volume within each minute as the day passes, and search for low-frequency traders
executing their massive orders on a chronological time-space. The largest concentra-
tions of volume within a minute tend to occur during the first few seconds, for almost
every hour of the day. This is particularly true at 00:00–01:00 GMT (around the open
of Asian markets), 05:00–09:00 GMT (around the open of U.K. and European equi-
ties), 13:00–15:00 GMT (around the open of U.S. equities), and 20:00–21:00 GMT
(around the close of U.S. equities).
A useful ML feature may be to evaluate the order imbalance at the beginning of
every minute, and determine whether there is a persistent component. This can then be
used to front-run large institutional investors, while the larger portion of their TWAP
order is still pending.


WHAT IS MICROSTRUCTURAL INFORMATION?
295
19.6.4
Options Markets
Muravyev et al. [2013] use microstructural information from U.S. stocks and
options to study events where the two markets disagree. They characterize such
disagreement by deriving the underlying bid-ask range implied by the put-call parity
quotes and comparing it to the actual bid-ask range of the stock. They conclude that
disagreements tend to be resolved in favor of stock quotes, meaning that option quotes
do not contain economically significant information. At the same time, they do find
that option trades contain information not included in the stock price. These findings
will not come as a surprise to portfolio managers used to trade relatively illiquid prod-
ucts, including stock options. Quotes can remain irrational for prolonged periods of
time, even as sparse prices are informative.
Cremers and Weinbaum [2010] find that stocks with relatively expensive calls
(stocks with both a high volatility spread and a high change in the volatility spread)
outperform stocks with relatively expensive puts (stocks with both a low volatility
spread and a low change in the volatility spread) by 50 basis points per week. This
degree of predictability is larger when option liquidity is high and stock liquidity
is low.
In line with these observations, useful features can be extracted from comput-
ing the put-call implied stock price, derived from option trades. Futures prices only
represent mean or expected future values. But option prices allow us to derive the
entire distribution of outcomes being priced. An ML algorithm can search for pat-
terns across the Greek letters quoted at various strikes and expiration dates.
19.6.5
Serial Correlation of Signed Order Flow
Toth et al. [2011] study the signed order flow of London Stock Exchange stocks,
and find that order signs are positively autocorrelated for many days. They attribute
this observation to two candidate explanations: Herding and order splitting. They
conclude that on timescales of less than a few hours, the persistence of order flow is
overwhelmingly due to splitting rather than herding.
Given that market microstructure theory attributes the persistency of order flow
imbalance to the presence of informed traders, it makes sense to measure the strength
of such persistency through the serial correlation of the signed volumes. Such a fea-
ture would be complementary to the features we studied in Section 19.5.
19.7
WHAT IS MICROSTRUCTURAL INFORMATION?
Let me conclude this chapter by addressing what I consider to be a major flaw in the
market microstructure literature. Most articles and books on this subject study asym-
metric information, and how strategic agents utilize it to profit from market makers.
But how is information exactly defined in the context of trading? Unfortunately, there
is no widely accepted definition of information in a microstructural sense, and the
literature uses this concept in a surprisingly loose, rather informal way (L´opez de
Prado [2017]). This section proposes a proper definition of information, founded on
signal processing, that can be applied to microstructural studies.


296
MICROSTRUCTURAL FEATURES
Consider a features matrix X = {Xt}t=1,…,T that contains information typically
used by market makers to determine whether they should provide liquidity at a par-
ticular level, or cancel their passive quotes. For example, the columns could be all of
the features discussed in this chapter, like VPIN, Kyle’s lambda, cancellation rates,
etc. Matrix X has one row for each decision point. For example, a market maker may
reconsider the decision to either provide liquidity or pull out of the market every
time 10,000 contracts are traded, or whenever there is a significant change in prices
(recall sampling methods in Chapter 2), etc. First, we derive an array y = {yt}t=1,…,T
that assigns a label 1 to an observation that resulted in a market-making profit, and
labels as 0 an observation that resulted in a market-making loss (see Chapter 3 for
labeling methods). Second, we fit a classifier on the training set (X, y). Third, as new
out-of-sample observations arrive 𝜏> T, we use the fit classifier to predict the label
̂y𝜏= E𝜏[y𝜏|X]. Fourth, we derive the cross-entropy loss of these predictions, L𝜏, as
described in Chapter 9, Section 9.4. Fifth, we fit a kernel density estimator (KDE) on
the array of negative cross-entropy losses, {−Lt}t=T+1,…,𝜏, to derive its cumulative
distribution function, F. Sixth, we estimate the microstructural information at time t
as 𝜙𝜏= F[−L𝜏], where 𝜙𝜏∈(0, 1).
This microstructural information can be understood as the complexity faced by
market makers’ decision models. Under normal market conditions, market makers
produce informed forecasts with low cross-entropy loss, and are able to profit from
providing liquidity to position takers. However, in the presence of (asymmetrically)
informed traders, market makers produce uninformed forecasts, as measured by high
cross-entropy loss, and they are adversely selected. In other words, microstructural
information can only be defined and measured relative to the predictive power of
market makers. The implication is that {𝜙𝜏} should become an important feature in
your financial ML toolkit.
Consider the events of the flash crash of May 6, 2010. Market makers wrongly
predicted that their passive quotes sitting on the bid could be filled and sold back at
a higher level. The crash was not caused by a single inaccurate prediction, but by the
accumulation of thousands of prediction errors (Easley et al. [2011]). If market mak-
ers had monitored the rising cross-entropy loss of their predictions, they would have
recognized the presence of informed traders and the dangerously rising probability
of adverse selection. That would have allowed them to widen the bid-ask spread to
levels that would have stopped the order flow imbalance, as sellers would no longer
have been willing to sell at those discounts. Instead, market makers kept providing
liquidity to sellers at exceedingly generous levels, until eventually they were forced to
stop-out, triggering a liquidity crisis that shocked markets, regulators, and academics
for months and years.
EXERCISES
19.1 From a time series of E-mini S&P 500 futures tick data,
(a) Apply the tick rule to derive the series of trade signs.
(b) Compare to the aggressor’s side, as provided by the CME (FIX tag 5797).
What is the accuracy of the tick rule?


EXERCISES
297
(c) Select the cases where FIX tag 5797 disagrees with the tick rule.
(i) Can you see anything distinct that would explain the disagreement?
(ii) Are these disagreements associated with large price jumps? Or high
cancelation rates? Or thin quoted sizes?
(iii) Are these disagreements more likely to occur during periods of high
or low market activity?
19.2 Compute the Roll model on the time series of E-mini S&P 500 futures tick
data.
(a) What are the estimated values of 𝜎2
u and c?
(b) Knowing that this contract is one of the most liquid products in the world,
and that it trades at the tightest possible bid-ask spread, are these values
in line with your expectations?
19.3 Compute the high-low volatility estimator (Section19.3.3.) on E-mini S&P 500
futures:
(a) Using weekly values, how does this differ from the standard deviation of
close-to-close returns?
(b) Using daily values, how does this differ from the standard deviation of
close-to-close returns?
(c) Using dollar bars, for an average of 50 bars per day, how does this differ
from the standard deviation of close-to-close returns?
19.4 Apply the Corwin-Schultz estimator to a daily series of E-mini S&P 500
futures.
(a) What is the expected bid-ask spread?
(b) What is the implied volatility?
(c) Are these estimates consistent with the earlier results, from exercises 2
and 3?
19.5 Compute Kyle’s lambda from:
(a) tick data.
(b) a time series of dollar bars on E-mini S&P 500 futures, where
(i) bt is the volume-weighted average of the trade signs.
(ii) Vt is the sum of the volumes in that bar.
(iii) Δpt is the change in price between two consecutive bars.
19.6 Repeat exercise 5, this time applying Hasbrouck’s lambda. Are results consis-
tent?
19.7 Repeat exercise 5, this time applying Amihud’s lambda. Are results consis-
tent?
19.8 Form a time series of volume bars on E-mini S&P 500 futures,
(a) Compute the series of VPIN on May 6, 2010 (flash crash).
(b) Plot the series of VPIN and prices. What do you see?
19.9 Compute the distribution of order sizes for E-mini S&P 500 futures
(a) Over the entire period.
(b) For May 6, 2010.


298
MICROSTRUCTURAL FEATURES
(c) Conduct a Kolmogorov-Smirnov test on both distributions. Are they sig-
nificantly different, at a 95% confidence level?
19.10 Compute a time series of daily quote cancellations rates, and the portion of
market orders, on the E-mini S&P 500 futures dataset.
(a) What is the correlation between these two series? Is it statistically signif-
icant?
(b) What is the correlation between the two series and daily volatility? Is this
what you expected?
19.11 On the E-mini S&P 500 futures tick data:
(a) Compute the distribution of volume executed within the first 5 seconds of
every minute.
(b) Compute the distribution of volume executed every minute.
(c) Compute the Kolmogorov-Smirnov test on both distributions. Are they
significantly different, at a 95% confidence level?
19.12 On the E-mini S&P 500 futures tick data:
(a) Compute the first-order serial correlation of signed volumes.
(b) Is it statistically significant, at a 95% confidence level?
REFERENCES
Abad, D. and J. Yague (2012): “From PIN to VPIN.” The Spanish Review of Financial Economics,
Vol. 10, No. 2, pp.74-83.
Aitken, M. and A. Frino (1996): “The accuracy of the tick test: Evidence from the Australian Stock
Exchange.” Journal of Banking and Finance, Vol. 20, pp. 1715–1729.
Amihud, Y. and H. Mendelson (1987): “Trading mechanisms and stock returns: An empirical inves-
tigation.” Journal of Finance, Vol. 42, pp. 533–553.
Amihud, Y. (2002): “Illiquidity and stock returns: Cross-section and time-series effects.” Journal of
Financial Markets, Vol. 5, pp. 31–56.
Andersen, T. and O. Bondarenko (2013): “VPIN and the Flash Crash.” Journal of Financial Markets,
Vol. 17, pp.1-46.
Beckers, S. (1983): “Variances of security price returns based on high, low, and closing prices.”
Journal of Business, Vol. 56, pp. 97–112.
Bethel, E. W., Leinweber. D., Rubel, O., and K. Wu (2012): “Federal market information technology
in the post–flash crash era: Roles for supercomputing.” Journal of Trading, Vol. 7, No. 2, pp.
9–25.
Carlin, B., M. Sousa Lobo, and S. Viswanathan (2005): “Episodic liquidity crises. Cooperative and
predatory trading.” Journal of Finance, Vol. 42, No. 5 (October), pp. 2235–2274.
Cheung, W., R. Chou, A. Lei (2015): “Exchange-traded barrier option and VPIN.” Journal of Futures
Markets, Vol. 35, No. 6, pp. 561-581.
Corwin, S. and P. Schultz (2012): “A simple way to estimate bid-ask spreads from daily high and
low prices.” Journal of Finance, Vol. 67, No. 2, pp. 719–760.
Cremers, M. and D. Weinbaum (2010): “Deviations from put-call parity and stock return predictabil-
ity.” Journal of Financial and Quantitative Analysis, Vol. 45, No. 2 (April), pp. 335–367.
Donefer, B. (2010): “Algos gone wild. Risk in the world of automated trading strategies.” Journal
of Trading, Vol. 5, pp. 31–34.
