# Centurion Capital — Business Plan & Company Setup Guide

*Created: 14 April 2026*

---

## 1. What You Have Today

| Layer | Built |
|-------|-------|
| **Signal Engine** | 23 forecast sources, Carver FDM combination, walk-forward optimizer |
| **Backtest** | 13-year PIT universe, slippage/cost modeling, Monte Carlo risk |
| **Risk Mgmt** | 7-layer drawdown protection, HMM regime, vol targeting, kill switch, distribution shift |
| **Live Trading** | Kite Connect, daily rebalancer, paper trader, circuit breaker |
| **Infra** | FastAPI + Next.js, PostgreSQL, MinIO, Redis, ChromaDB, Sentry, CI/CD |
| **Research** | RAG pipeline, news scraping, sentiment, signal validation framework |

**Performance:** Sharpe 1.127, CAGR 30.4%, MaxDD 29.6% (backtest). Realistic live estimate: Sharpe ~1.26, CAGR ~25%.

**Honest flags:** Detrended Sharpe ≈ 0 (mostly market β), struggles in flat markets (WF1: -0.22), no live track record yet.

---

## 2. Business Models Available

| Model | SEBI Registration | Min Capital | Time to Revenue | Revenue Potential |
|-------|------------------|-------------|----------------|-------------------|
| **A. Prop Trading** | None | ₹5-50L personal | Immediate | Limited by capital |
| **B. Research Analyst** | RA registration (₹1L NTA) | ₹5-10L | 3-6 months | ₹5-15L/yr (signals) |
| **C. RIA (Advisory)** | RIA registration (₹50L NTA) | ₹50L+ | 6-12 months | ₹50L-1Cr/yr |
| **D. PMS** | PMS registration (₹2Cr NW) | ₹2Cr | 12-18 months | ₹1Cr+/yr |
| **E. AIF Cat III** | AIF registration | ₹20Cr corpus | 18-24 months | ₹5Cr+/yr |
| **F. SaaS Platform** | None (pure software) | ₹5-20L | 3-6 months | ₹20L-1Cr/yr |

---

## 3. Recommended Path: Phased Approach

### Phase 1 — Prop Trade + Validate (Month 0-6)

- Trade **₹5L personal capital** via Zerodha (sole proprietor)
- Complete paper trading → staged live (₹1L → 2L → 3L → 4L → 5L)
- **Build auditable 6-month track record** (Sharpe ≥ 0.60 target)
- No registration needed — you're a retail trader

### Phase 2 — Company + Scale Capital (Month 6-12)

- **Incorporate LLP** (Centurion Capital LLP)
  - Cost: ~₹25K, Time: 2 weeks
  - 30% flat tax on trading profits
  - Lower compliance than Pvt Ltd
- Bring capital to ₹25-50L (personal + close network)
- Build public track record dashboard
- Launch signal API as paid product

### Phase 3 — Revenue Diversification (Month 12-24)

| Stream | Monthly Revenue |
|--------|----------------|
| Prop trading (₹50L @ 25% CAGR) | ₹73K/mo (after tax) |
| Signal API (50 subs × ₹10K) | ₹5L/mo |
| SEBI RA advisory (100 clients × ₹1L/yr) | ₹8.3L/mo |
| **Total** | **₹14L/mo** |

### Phase 4 — Institutional (Month 24+)

- PMS registration (if ₹2Cr NW achieved) → ₹50L+ per client
- Or AIF Cat III → hedge fund structure
- 2% mgmt + 20% perf fee on ₹100Cr AUM = ₹2Cr+/yr

---

## 4. Company Structure

### LLP (Recommended start)

**Why LLP over Pvt Ltd:**
- No minimum capital. 30% flat tax (no dividend complications)
- Flexible profit sharing. Lower compliance (~₹50K/yr audit + filings)
- Easy conversion to Pvt Ltd later when raising equity

**Registration steps:**
1. DSC (₹1,500, 2 days)
2. DIN (free with SPICe+, 1 day)
3. Reserve name on MCA (₹200, 2 days)
4. File FiLLiP form (₹500, 5-7 days)
5. LLP Agreement (₹5K-15K, lawyer)
6. PAN/TAN (auto), Bank account, GST

**Total: ₹15-25K | 10-15 days**

### Convert LLP → Pvt Ltd when:
- Issuing equity to investors (ESOPs, angel)
- AUM > ₹5 Cr
- Revenue > ₹1 Cr/yr

---

## 5. SEBI Regulatory Summary

**No registration needed for:**
- Trading your own money via broker API ✅
- Selling software tools (no investment advice) ✅
- Backtesting, research, educational content ✅

**Registration required for:**
- Selling buy/sell signals → **Research Analyst (RA)** — lightest option (₹1L NTA, NISM XV exam)
- Personalized advice → **RIA** (₹50L NTA for non-individual)
- Managing others' portfolios → **PMS** (₹2Cr NW)
- Pooling investor funds → **AIF** (₹20Cr corpus)

**Tax:** LLP trading profits = 30% flat. Maintain books (mandatory if turnover > ₹1 Cr). Quarterly advance tax.

---

## 6. Capital vs Income Reality Check

| Capital | Net CAGR | Annual Net (after 30% tax) | Monthly |
|---------|---------|---------------------------|---------|
| ₹5L | 25% | ₹87K | ₹7.3K |
| ₹25L | 25% | ₹4.4L | ₹37K |
| ₹50L | 25% | ₹8.75L | ₹73K |
| ₹1Cr | 25% | ₹17.5L | ₹1.46L |

**Hard truth:** ₹5-10L alone won't make a living. Need either ₹50L+ capital OR supplementary product revenue.

---

## 7. Technology Productization (to sell as SaaS/API)

| Priority | What | Effort |
|----------|------|--------|
| 1 | **Broker abstraction** (add Upstox, Angel One, Dhan) | 4-6 weeks |
| 2 | **Dashboard upgrade** (real-time P&L, signals, risk) | 6-8 weeks |
| 3 | **Public API + auth** (rate-limited, API keys) | 3-4 weeks |
| 4 | **Multi-tenancy** (per-tenant config/data) | 4-6 weeks |
| 5 | **Billing** (Razorpay integration) | 2 weeks |

**Total MVP: ~6-8 months solo**

---

## 8. Risk Matrix

| Risk | Mitigation |
|------|-----------|
| Strategy stops working | Distribution shift detector, regime-adaptive sizing, multi-strategy diversification |
| Flat market (like 2018-19) | Bear defense already shrinks positions. Keep 12mo expenses in FD |
| SEBI tightens algo rules | Get RA/RIA registration proactively. Don't operate in grey areas |
| Broker API shutdown | Multi-broker abstraction |
| Cash flow gap Year 1 | **Don't quit day job until 6mo live track record + ₹50K/mo product revenue** |

---

## 9. 12-Month Execution Checklist

**Month 1-2:** Complete paper trading, begin live ₹1L, register sole prop, open trading bank account

**Month 3-4:** Scale to ₹3L, set up daily email reports, build track record page, file trademark

**Month 5-6:** Incorporate LLP, transfer to LLP demat, GST, hire CA, deploy full ₹5L

**Month 7-8:** Build broker abstraction + dashboard, create signal API, beta test 5-10 users

**Month 9-10:** Launch paid API, NISM XV cert, apply SEBI RA, target 10 paying customers

**Month 11-12:** Review 6mo live track record → **If ₹50K+/mo: go full-time. If not: iterate.**

---

## 10. Competitive Edge

Most Indian algo platforms (Streak, Tradetron, AlgoTest) offer simple technical indicators. Your differentiation:
- **23 proprietary signals** with academic rigor (Carver, FDM, PBO)
- **Institutional-grade risk**: 7-layer drawdown, regime detection, distribution shift
- **Full stack**: signal engine + platform + live execution (competitors are usually one of these)
- **Survivorship-bias-free backtesting** — rare in Indian market

---

## Bottom Line

```
The single most important thing right now:
  Paper trading → Live → 6-month verifiable track record

Everything else (company, product, clients) depends on this.
Don't quit your job until you have both:
  (a) 6 months of live Sharpe ≥ 0.60, AND
  (b) ₹50K+/month revenue (trading + product combined)
```

---
---

# PART 2: Proprietary Trading Playbook (Deep Dive)

---

## 11. Executive Summary — Prop Trading

Proprietary trading (prop trading) means trading your own capital for profit.
No clients, no SEBI registration, no AUM fees. Pure alpha extraction.

This is the **fastest path to revenue** from centurion_core because:
- Zero regulatory overhead (you're a retail trader)
- Infrastructure is 95% built (Kite Connect, rebalancer, risk layers)
- Revenue starts from Day 1 of live trading
- Track record builds automatically for future business expansion

---

## 12. Infrastructure Audit — What's Ready

### Trading Engine — READY

| Component | File | Status |
|-----------|------|--------|
| Kite Connect integration | `kite_connect/core/kite_client.py` | ✅ Built |
| Order placement (market/limit/SL) | `kite_connect/trading/order_manager.py` | ✅ Built |
| Daily rebalancer | `kite_connect/trading/daily_rebalancer.py` | ✅ Built |
| Paper trader | `kite_connect/trading/paper_trader.py` | ✅ Built |
| Position tracker | `kite_connect/trading/position_tracker.py` | ✅ Built |
| Portfolio snapshot | `kite_connect/trading/portfolio_snapshot.py` | ✅ Built |

### Signal Engine — READY

| Component | Status |
|-----------|--------|
| 23 active forecast signals | ✅ Weighted, combined via FDM |
| Walk-forward optimizer | ✅ 5-fold validated |
| Forecast combiner | ✅ Auto-renormalization |
| Signal quality evaluator | ✅ Per-signal Sharpe tracking |

### Risk Management — READY

| Layer | Implementation | Status |
|-------|---------------|--------|
| 1. Vol targeting | 20% annualized, EWMA σ | ✅ |
| 2. Position limits | MAX_POSITIONS=15, per-symbol caps | ✅ |
| 3. Stop losses | ATR-based trailing stops | ✅ |
| 4. Regime detection | HMM + SMA-based, 4 states | ✅ |
| 5. Drawdown circuit breaker | 3-tier (10%/15%/20%) | ✅ |
| 6. Kill switch | Manual override, auto-trigger | ✅ |
| 7. Distribution shift detector | KS-test on returns | ✅ |

### Monitoring — READY

| Component | Status |
|-----------|--------|
| Email alerts | ✅ `services/email_service.py` |
| Sentry error tracking | ✅ Configured |
| Daily P&L reports | ✅ `reports/daily_report.py` |
| Risk dashboard | ✅ Next.js frontend |

### What's NOT Ready

| Gap | Impact | Fix Effort |
|-----|--------|------------|
| No live track record | Can't validate real-world Sharpe | 6 months of live trading |
| Single broker (Zerodha only) | Single point of failure | 4-6 weeks (add Upstox/Angel) |
| No automated token refresh | Kite token expires daily, needs manual login | 1 week (implement totp auto-login) |
| No mobile alerts | Can't monitor away from desk | 2 days (Telegram bot) |

---

## 13. Capital Plan

### Starting Capital: ₹5,00,000

This is the minimum for meaningful signal testing across 15 position slots.

| Per-position allocation | ₹33,333 |
|------------------------|---------|
| Min share price to trade | ~₹50 (can buy 666 shares) |
| Max share price practical | ~₹5,000 (can buy 6 shares — too few) |
| Sweet spot universe | Stocks ₹100-₹2,000 |

### Capital Scaling Plan

| Stage | Capital | Monthly Net (25% CAGR, 30% tax) | Trigger to Advance |
|-------|---------|--------------------------------|---------------------|
| **Paper** | ₹0 | ₹0 | 30 days, Sharpe ≥ 0.60 |
| **Stage 1** | ₹1,00,000 | ₹1,458 | 60 days, no single day > -3% |
| **Stage 2** | ₹3,00,000 | ₹4,375 | 90 days, Sharpe ≥ 0.50 live |
| **Stage 3** | ₹5,00,000 | ₹7,292 | 6 months, Sharpe ≥ 0.60 live |
| **Stage 4** | ₹10,00,000 | ₹14,583 | 9 months, MaxDD < 20% |
| **Stage 5** | ₹25,00,000 | ₹36,458 | 12 months, consistent profitability |
| **Stage 6** | ₹50,00,000 | ₹72,917 | 18 months, track record auditable |
| **Stage 7** | ₹1,00,00,000 | ₹1,45,833 | 24 months, ready for external capital |

### Capital Sources (Beyond Personal Savings)

| Source | Amount | Terms | When |
|--------|--------|-------|------|
| Personal savings | ₹5-10L | No cost | Now |
| Family & friends | ₹10-25L | Profit-sharing, no guarantee | After 6mo track record |
| HNI informal pool | ₹25L-1Cr | 20% profit share | After 12mo track record |
| Angel investment (in tech) | ₹50L-2Cr | Equity in LLP/Pvt Ltd | After product revenue proven |

**WARNING:** Taking others' money without SEBI registration = illegal.
Family/friends giving you money to trade "on their behalf" is a grey area.
Safest: they open their own Zerodha accounts, you provide signals (get RA license).

---

## 14. Daily Operations Protocol

### Pre-Market (8:30 AM - 9:00 AM)

```
1. Check overnight global cues (US markets, SGX Nifty, Asia)
2. Verify Kite login token is valid
3. Review email report from previous day's close
4. Check kill switch status — ensure system is armed
5. Review pending orders (AMO placed previous evening)
```

### Market Hours (9:15 AM - 3:30 PM)

```
1. 9:15 — Opening auction. AMO orders execute.
2. 9:30 — Verify all fills. Log slippage vs expected price.
3. 10:00 — Check if any stop-loss triggers fired.
4. 12:00 — Midday health check (positions, margin, P&L).
5. 3:00 — Pre-close review. System computes new signals.
6. 3:20 — Place AMO orders for next day (new entries/exits).
7. 3:30 — Market close. Final snapshot saved.
```

### Post-Market (3:30 PM - 4:30 PM)

```
1. Run signal engine on closing prices
2. Generate daily report (P&L, risk metrics, signal quality)
3. Email report to self
4. Review any distribution shift alerts
5. Update trade journal (manual notes on unusual events)
6. Place AMO orders for next morning
```

### Weekly (Sunday Evening)

```
1. Review weekly performance vs benchmark (Nifty500 TRI)
2. Check signal degradation (rolling 60-day per-signal Sharpe)
3. Review regime state history
4. Backup database and logs
5. Check for yfinance/Kite API changes or deprecations
```

### Monthly

```
1. Full performance report (Sharpe, CAGR, MaxDD, Sortino, Calmar)
2. Compare live vs backtest expectations — flag divergence > 1σ
3. Tax provisioning (30% of net profits → separate account)
4. Rebalance signal weights if walk-forward window suggests change
5. Infrastructure maintenance (update packages, rotate API keys)
```

---

## 15. Risk Limits — Non-Negotiable Rules

### Hard Limits (Kill Switch Triggers)

| Rule | Threshold | Action |
|------|-----------|--------|
| Daily loss | > 3% of equity | Halt all trading for 24 hours |
| Weekly loss | > 5% of equity | Reduce position count to 5 for 1 week |
| Monthly loss | > 10% of equity | Reduce to 3 positions + manual review |
| Drawdown from peak | > 20% | Kill switch: flatten all, pause 2 weeks |
| Single position loss | > 8% of position value | Auto stop-loss (already implemented) |
| Regime: severe_bear | Detected | Floor at 10% exposure, min 1 position |

### Soft Limits (Alerts Only)

| Rule | Threshold | Action |
|------|-----------|--------|
| Correlation spike | > 0.7 avg pairwise | Email alert, review diversification |
| Signal degradation | Any signal 60d Sharpe < -0.5 | Flag for weight review |
| Turnover spike | > 50% daily | Check for data issues |
| Execution slippage | > 1% avg | Review order type (market → limit) |

### Position Concentration

| Constraint | Limit |
|-----------|-------|
| Max positions | 15 |
| Max per sector | 40% of capital |
| Max single stock | 15% of capital |
| Min positions (severe_bear) | 1 |
| Long-only enforcement | YES (IND market) |

---

## 16. Execution Strategy — Zerodha Kite Specifics

### Order Types

| Scenario | Order Type | Why |
|----------|-----------|-----|
| New entry (next day) | AMO-LIMIT at prev close ± 0.5% | Avoids market open slippage |
| Stop loss | SL-M (stop-loss market) | Guaranteed exit, accept slippage |
| Full exit (signal flip) | AMO-LIMIT at prev close ± 0.3% | |
| Partial resize | Regular LIMIT during market hours | Can wait for fill |

### Zerodha Cost Structure (Equity Delivery)

| Component | Rate | Per ₹1L Trade |
|-----------|------|---------------|
| Brokerage | ₹0 | ₹0 |
| STT | 0.1% (sell) | ₹100 |
| Exchange txn | 0.00345% | ₹3.45 |
| GST | 18% on exchange txn | ₹0.62 |
| Stamp duty | 0.015% (buy) | ₹15 |
| SEBI | 0.0001% | ₹0.10 |
| **Round-trip total** | **~0.12%** | **~₹119** |

With 15 positions turning over ~40x/year each:
- Annual trades: ~600 round-trips
- Average trade size: ₹33K (at ₹5L capital)
- **Annual transaction cost: ~₹24,000 (4.8% of capital)**

This is already modeled in the backtest (Phase 2 cost tiering).

### Token Management

Kite API tokens expire daily. Options:
1. **Manual:** Login via browser each morning (current approach)
2. **TOTP auto-login:** Use `pyotp` + Kite's TOTP flow (build this — 1 week)
3. **Kite Publisher:** Webhook-based, but limited to certain plans

**Recommendation:** Build TOTP auto-login before going live. Manual login
will break the system if you oversleep or travel.

---

## 17. Tax & Compliance

### Tax Treatment

| Income Type | Tax Rate | Classification |
|------------|---------|----------------|
| Short-term capital gains (equity, held < 12mo) | 20% | Most trades will be STCG |
| Long-term capital gains (equity, held > 12mo) | 12.5% (above ₹1.25L) | Unlikely with daily rebalancing |
| Speculative income (intraday) | Slab rate (up to 30%) | Not applicable — system is delivery-based |
| **Effective rate for algo trading** | **20% STCG** | All positions are delivery, most held < 12 months |

**Note:** STCG changed to 20% in Budget 2024. Verify current rate with CA.

### Record Keeping Requirements

| Document | Frequency | Tool |
|----------|-----------|------|
| Trade log (entry/exit/P&L) | Every trade | Auto-generated by system |
| Portfolio snapshot | Daily | Auto-saved to PostgreSQL |
| P&L statement | Monthly | Generate from trade log |
| Capital gains report | Annual | Zerodha provides, cross-verify |
| ITR filing | Annual | CA required (ITR-3 for trading income) |

### Audit Trail

The system already logs everything to PostgreSQL + MinIO.
Ensure these are backed up monthly to a separate location.
Keep logs for **8 years** (Income Tax Act requirement).

---

## 18. Psychological Preparation

### What Will Happen (Guaranteed)

| Event | When | Your Reaction | Correct Response |
|-------|------|--------------|-----------------|
| 5 losing days in a row | Month 1-2 | Panic, want to override system | Do nothing. Check if within risk limits. |
| System holds cash while market rallies | First bear detection | FOMO, want to disable regime filter | Do nothing. The filter saved you from crashes. |
| -15% drawdown from peak | Within first year | Doubt everything, want to quit | Check if DD is within backtest expectations (30-38%). |
| A single stock gaps down -10% overnight | Random | Blame the signal, want to tweak weights | Check if stop-loss fired. One bad trade ≠ broken system. |
| Friend's random stock pick outperforms you | Constantly | Question why you built all this | Compare risk-adjusted returns, not raw returns. |

### Rules for Yourself

1. **Never override the system during market hours.** If you want to change something, do it over the weekend after reviewing data.
2. **Never increase capital after a winning streak.** Follow the staged capital plan.
3. **Never decrease capital after a losing streak** (unless kill switch triggers). Mean reversion works for your system too.
4. **Keep a trading journal.** Write 3 lines every evening: what happened, how you felt, what you'd do differently. This is your most valuable asset for Year 2.
5. **Take 1 week off every quarter.** The system runs itself. If you can't walk away for a week, the automation isn't ready for live.

---

## 19. Benchmarking & Performance Tracking

### Benchmark: NIFTY500 TRI (Total Return Index)

Do NOT compare against NIFTY50. Your universe is NIFTY500.
TRI includes dividends — fair comparison for a system that captures carry.

### Monthly Scorecard Template

```
Month: ___________
Starting Equity: ₹___________
Ending Equity:   ₹___________
Return:          ___%
Benchmark:       ___%
Alpha:           ___%

Sharpe (rolling 60d):  ___
MaxDD (rolling 60d):   ___%
Win Rate:              ___%
Avg Win / Avg Loss:    ___

Positions Held (avg):  ___
Turnover:              ___%
Total Costs:           ₹___

Regime Distribution:
  Strong Bull: ___% of days
  Mild Bull:   ___% of days
  Mild Bear:   ___% of days
  Severe Bear: ___% of days

Signal Health:
  Best signal (60d Sharpe):  ___ (_____)
  Worst signal (60d Sharpe): ___ (_____)
  Any degradation alerts:    Y/N

Notes:
_________________________________
```

---

## 20. Go-Live Checklist

### Before Paper Trading

- [ ] All 7 risk layers verified in unit tests
- [ ] Kill switch tested (manual trigger + auto trigger)
- [ ] Email alerts verified (test email received)
- [ ] Daily rebalancer produces correct orders for 5 manual scenarios
- [ ] Paper trader matches backtest output for 10 random days

### Before Stage 1 (₹1L Live)

- [ ] 30 days paper trading completed
- [ ] Paper Sharpe ≥ 0.60
- [ ] No single paper day loss > 3%
- [ ] Kite TOTP auto-login working
- [ ] Trade journal started
- [ ] CA consulted on tax treatment
- [ ] Emergency fund (6 months expenses) separate from trading capital

### Before Stage 3 (₹5L Live)

- [ ] 6 months live track record
- [ ] Live Sharpe ≥ 0.50
- [ ] MaxDD < 25% in live
- [ ] Survived at least 1 regime change
- [ ] Monthly tax provisioning in place
- [ ] Backup broker account opened (Upstox/Angel)

### Before Stage 5 (₹25L Live)

- [ ] 12 months live track record
- [ ] Live Sharpe ≥ 0.60
- [ ] LLP incorporated
- [ ] Professional CA engaged
- [ ] Quarterly advance tax being paid
- [ ] Track record dashboard public
- [ ] Considered SEBI RA registration

---

## 21. Financial Projections — Conservative

Assumptions: 25% gross CAGR, 20% STCG tax, 5% transaction costs,
2% infrastructure costs (server, data, API)

| Year | Capital (Start) | Gross Return | Costs | Tax | Net Return | Capital (End) |
|------|----------------|-------------|-------|-----|-----------|--------------|
| 1 | ₹5,00,000 | ₹1,25,000 | ₹35,000 | ₹18,000 | ₹72,000 | ₹5,72,000 |
| 2 | ₹10,00,000 | ₹2,50,000 | ₹70,000 | ₹36,000 | ₹1,44,000 | ₹11,44,000 |
| 3 | ₹25,00,000 | ₹6,25,000 | ₹1,75,000 | ₹90,000 | ₹3,60,000 | ₹28,60,000 |
| 4 | ₹50,00,000 | ₹12,50,000 | ₹3,50,000 | ₹1,80,000 | ₹7,20,000 | ₹57,20,000 |
| 5 | ₹1,00,00,000 | ₹25,00,000 | ₹7,00,000 | ₹3,60,000 | ₹14,40,000 | ₹1,14,40,000 |

**Year 1 net: ₹6,000/month** — not a living.
**Year 3 net: ₹30,000/month** — supplementary income.
**Year 5 net: ₹1,20,000/month** — viable full-time income.

Capital jumps in Years 2-4 assume additional capital injection
(savings, family, or product revenue reinvested).

---

## 22. What Makes This a Business (Not Just Trading)

| Asset | Value | Monetizable? |
|-------|-------|-------------|
| Signal engine IP | 23 signals, FDM combiner, PBO validation | Yes — license/SaaS |
| Risk framework | 7-layer institutional-grade | Yes — white-label |
| Backtest engine | PIT universe, realistic costs | Yes — SaaS tool |
| Track record | Auditable, timestamped | Yes — attracts capital |
| Codebase | 50K+ lines, production-grade | Yes — acqui-hire target |
| Domain expertise | Indian markets + quant methods | Yes — consulting |

**The prop trading is the proof.** The technology is the product.
Trade to prove it works. Sell the technology to scale.
