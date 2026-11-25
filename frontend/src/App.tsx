import { useState } from 'react'
import './App.css'

type MetricKey = 'silhouette' | 'dbi' | 'runtime'

const metricOptions: {
  key: MetricKey
  label: string
  direction: 'higher' | 'lower'
  description: string
}[] = [
  {
    key: 'silhouette',
    label: 'Silhouette',
    direction: 'higher',
    description: 'Higher is better: a stronger balance of within-group similarity and separation.',
  },
  {
    key: 'dbi',
    label: 'Davies–Bouldin',
    direction: 'lower',
    description: 'Lower is better: more compact groups relative to their separation.',
  },
  {
    key: 'runtime',
    label: 'Runtime',
    direction: 'lower',
    description: 'Lower is faster in this run. Timing depends on the implementation and machine.',
  },
]

const methods = [
  {
    number: '01',
    name: 'K-Means',
    type: 'Centroid-based',
    description:
      'Partitions projects into four groups, repeatedly assigning each project to its nearest center and recalculating the centers.',
    result: '4 clusters',
  },
  {
    number: '02',
    name: 'K-Medoids',
    type: 'Medoid-based · PAM',
    description:
      'Builds groups around representative projects selected from the dataset, then improves them through medoid swaps.',
    result: '4 clusters',
  },
  {
    number: '03',
    name: 'DBSCAN',
    type: 'Density-based',
    description:
      'Finds dense groups without choosing the cluster count in advance, and marks isolated projects as noise.',
    result: '4 clusters · 47 noise',
  },
]

const features = [
  { name: 'Budget', description: 'Contract cost in Philippine pesos', icon: '₱' },
  { name: 'Duration', description: 'Calendar days from start to completion', icon: '◷' },
  { name: 'Contractor count', description: 'Projects handled by the contractor', icon: '▦' },
  { name: 'Average budget', description: 'Mean project budget per contractor', icon: '↗' },
  { name: 'Delay rate', description: 'Share marked terminated or missing completion', icon: '%' },
]

const results = [
  {
    method: 'K-Means',
    clusters: 4,
    noise: null,
    silhouette: 0.3888,
    dbi: 1.0020,
    runtime: 0.0124,
  },
  {
    method: 'K-Medoids',
    clusters: 4,
    noise: null,
    silhouette: 0.2621,
    dbi: 1.2323,
    runtime: 30.9019,
  },
  {
    method: 'DBSCAN',
    clusters: 4,
    noise: 47,
    silhouette: 0.3601,
    dbi: 0.7818,
    runtime: 9.2070,
  },
]

function App() {
  const comparisonImage = `${import.meta.env.BASE_URL}methods_comparison.png`
  const [activeMetric, setActiveMetric] = useState<MetricKey>('silhouette')
  const selectedMetric = metricOptions.find((metric) => metric.key === activeMetric) ?? metricOptions[0]
  const rankedResults = [...results].sort((first, second) =>
    selectedMetric.direction === 'higher'
      ? second[activeMetric] - first[activeMetric]
      : first[activeMetric] - second[activeMetric],
  )
  const leadingMethod = rankedResults[0]

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="LEAD-AI home">
          <span className="brand-mark">L</span>
          <span>LEAD<span className="brand-accent">·</span>AI</span>
        </a>
        <nav className="site-nav" aria-label="Main navigation">
          <a href="#project">Project</a>
          <a href="#data">Dataset</a>
          <a href="#methods">Methods</a>
          <a href="#results">Results</a>
        </nav>
        <a
          className="nav-cta"
          href="https://github.com/randomchaffee/lead-ai"
          target="_blank"
          rel="noreferrer"
        >
          View source <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-orb hero-orb-one" aria-hidden="true" />
        <div className="hero-orb hero-orb-two" aria-hidden="true" />
        <div className="hero-inner">
          <div className="hero-copy">
            <p className="eyebrow"><span /> CMSC 170</p>
            <h1>Making sense of<br /><em>public works.</em></h1>
            <p className="hero-lede">
              Our project aims to discover project patterns and potential anomalies in Philippine
              infrastructure through three from-scratch clustering methods.
            </p>
            <div className="hero-actions">
              <a className="button button-light" href="#results">Results<span>↓</span></a>
              <a className="button button-outline" href="#methods">Methods</a>
            </div>
            <p className="hero-note">A student research project · UPLB </p>
          </div>
          <div className="hero-visual" aria-label="Project dataset overview">
            <div className="visual-topline">
              <span className="visual-year">2021 — 2023</span>
            </div>
            <div className="visual-number">1,600</div>
            <div className="visual-label">sampled infrastructure projects</div>
            <div className="visual-chart" aria-hidden="true">
              <span style={{ left: '12%', bottom: '17%', background: '#7bd5bd' }} />
              <span style={{ left: '19%', bottom: '24%', background: '#f6c873' }} />
              <span style={{ left: '25%', bottom: '33%', background: '#8bb7e6' }} />
              <span style={{ left: '31%', bottom: '20%', background: '#7bd5bd' }} />
              <span style={{ left: '37%', bottom: '44%', background: '#f6c873' }} />
              <span style={{ left: '43%', bottom: '31%', background: '#8bb7e6' }} />
              <span style={{ left: '49%', bottom: '57%', background: '#7bd5bd' }} />
              <span style={{ left: '55%', bottom: '36%', background: '#f6c873' }} />
              <span style={{ left: '61%', bottom: '48%', background: '#8bb7e6' }} />
              <span style={{ left: '68%', bottom: '68%', background: '#7bd5bd' }} />
              <span style={{ left: '75%', bottom: '53%', background: '#f6c873' }} />
              <span style={{ left: '82%', bottom: '77%', background: '#8bb7e6' }} />
              <span style={{ left: '90%', bottom: '62%', background: '#7bd5bd' }} />
            </div>
            <div className="visual-axis"><span>LOWER BUDGET</span><span>HIGHER BUDGET</span></div>
            <div className="visual-footer">
              <span><i className="legend-dot mint" /> Budget</span>
              <span><i className="legend-dot gold" /> Duration</span>
              <span className="visual-caption">*This is Illustrative</span>
            </div>
          </div>
        </div>
        <a className="scroll-cue" href="#project" aria-label="Scroll to project overview">SCROLL TO EXPLORE<span>↓</span></a>
      </section>

      <section className="intro-section content-section" id="project">
        <div className="section-heading">
          <p className="section-kicker">THE PROJECT</p>
          <h2>Looking For Patterns in DPWH Projects.<br /><span>Using AI Algorithms.</span></h2>
        </div>
        <div className="intro-copy">
          <p>
            Public infrastructure projects are complex. Our project, LEAD-AI, explores whether
            clustering can help organize DPWH contract records into groups with
            similar budgets, timelines, and contractor characteristics.
          </p>
          <p>
            The goal is to seek out patterns and unusual cases for closer review.
          </p>
        </div>
      </section>

      <section className="data-section" id="data">
        <div className="content-section data-inner">
          <div className="section-heading">
            <p className="section-kicker">01 / THE DATA</p>
            <h2>From raw records<br />to useful signals.</h2>
            <p className="section-description">
              Our Source: DPWH Infrastructure Transparency Dataset ·{' '}
              <a
                className="source-link"
                href="https://data.bettergov.ph/"
                target="_blank"
                rel="noreferrer"
              >
                BetterGov.ph ↗
              </a>
            </p>
          </div>
          <div className="data-facts">
            <div className="fact-card fact-primary">
              <span className="fact-label">SOURCE DATA</span>
              <strong>248,220</strong>
              <span>infrastructure contract records</span>
            </div>
            <div className="fact-card">
              <span className="fact-label">YEAR FILTER</span>
              <strong>2021–23</strong>
              <span>project start dates</span>
            </div>
            <div className="fact-card">
              <span className="fact-label">CONTRACTOR FILTER</span>
              <strong>3+ projects</strong>
              <span>minimum track record</span>
            </div>
            <div className="fact-card">
              <span className="fact-label">SAMPLE</span>
              <strong>1,600</strong>
              <span>randomly selected · seed 42</span>
            </div>
          </div>
        </div>
        <div className="features-wrap">
          <div className="feature-heading">
            <p className="section-kicker">FIVE FEATURES</p>
            <p>We used five distinct features from each project .</p>
          </div>
          <div className="feature-list">
            {features.map((feature, index) => (
              <article className="feature-item" key={feature.name}>
                <span className="feature-index">0{index + 1}</span>
                <span className="feature-icon" aria-hidden="true">{feature.icon}</span>
                <div><h3>{feature.name}</h3><p>{feature.description}</p></div>
              </article>
            ))}
          </div>
          <p className="normalization-note">
            All five features were Min-Max normalized to <strong>0–1</strong> so they can be
            compared on a common scale. Contractor statistics use the sampled records.
          </p>
        </div>
      </section>

      <section className="methods-section content-section" id="methods">
        <div className="section-heading heading-inline">
          <div>
            <p className="section-kicker">02 / THREE APPROACHES</p>
            <h2>Three methods to<br /><span>find a pattern.</span></h2>
          </div>
          <p className="section-description">
            We used three different methods to evaluate our chosen dataset. 
          </p>
        </div>
        <div className="method-grid">
          {methods.map((method, index) => (
            <article className={`method-card method-card-${index + 1}`} key={method.name}>
              <div className="method-card-top"><span>{method.number}</span><span>{method.type}</span></div>
              <div className="method-symbol" aria-hidden="true">
                {index === 0 ? '◎' : index === 1 ? '⌖' : '⠿'}
              </div>
              <h3>{method.name}</h3>
              <p>{method.description}</p>
              <div className="method-output"><span>THIS RUN</span><strong>{method.result}</strong></div>
            </article>
          ))}
        </div>
        <p className="method-footnote">
          K-Means and K-Medoids use k = 4. DBSCAN uses eps = 0.15 and minPts = 5.
          Each method assigns its own arbitrary cluster labels.
        </p>
      </section>

      <section className="results-section" id="results">
        <div className="content-section results-inner">
          <div className="results-heading">
            <div>
              <p className="section-kicker">03 / WHAT WE FOUND</p>
              <h2>Results Overview.</h2>
            </div>
          </div>
          <div className="metric-highlights">
            <article className="metric-card">
              <span className="metric-icon">↑</span>
              <div><span className="metric-label">TOP SILHOUETTE</span><strong>0.3888</strong><p>K-Means · higher is better</p></div>
            </article>
            <article className="metric-card">
              <span className="metric-icon metric-icon-blue">↓</span>
              <div><span className="metric-label">LOWEST DAVIES–BOULDIN</span><strong>0.7818</strong><p>DBSCAN · lower is better*</p></div>
            </article>
            <article className="metric-card">
              <span className="metric-icon metric-icon-orange">⌁</span>
              <div><span className="metric-label">DBSCAN NOISE POINTS</span><strong>47</strong></div>
            </article>
          </div>
          <div className="metric-explainer">
            <p><strong>Silhouette</strong> summarizes how well each project fits its group versus nearby groups. Higher is better.</p>
            <p><strong>Davies–Bouldin</strong> compares how compact and separated the groups are. Lower is better.</p>
          </div>
          <div className="table-shell">
            <div className="table-title">
              <div><h3>Method comparison</h3><span>Evaluation on normalized features</span></div>
              <div className="metric-switch" role="group" aria-label="Rank methods by metric">
                {metricOptions.map((metric) => (
                  <button
                    className={activeMetric === metric.key ? 'metric-option is-active' : 'metric-option'}
                    key={metric.key}
                    type="button"
                    aria-pressed={activeMetric === metric.key}
                    onClick={() => setActiveMetric(metric.key)}
                  >
                    {metric.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="ranking-note" aria-live="polite">
              <span className="ranking-badge">LEADING BY {selectedMetric.label.toUpperCase()}</span>
              <strong>{leadingMethod.method}</strong>
              <span>{selectedMetric.description}</span>
            </div>
            <div className="table-scroll">
              <table>
                <thead><tr>
                  <th scope="col">Method</th><th scope="col">Clusters</th><th scope="col">Noise</th>
                  <th scope="col">Silhouette <span>↑</span></th><th scope="col">Davies–Bouldin <span>↓</span></th><th scope="col">Runtime</th>
                </tr></thead>
                <tbody>
                  {rankedResults.map((row, index) => (
                    <tr className={index === 0 ? 'top-ranked' : undefined} key={row.method}>
                      <td><span className={`table-dot dot-${row.method.toLowerCase().replace('-', '')}`} />{row.method}{index === 0 && <span className="best-tag">TOP BY {selectedMetric.label.toUpperCase()}</span>}</td>
                      <td>{row.clusters}</td><td>{row.noise ?? '—'}</td>
                      <td className="numeric">{row.silhouette.toFixed(4)}</td>
                      <td className="numeric">{row.dbi.toFixed(4)}</td>
                      <td>{row.runtime.toFixed(4)} s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="table-footnote">* DBSCAN’s Silhouette and Davies–Bouldin scores exclude its 47 noise points. Runtime depends on this implementation and machine.</p>
          </div>
        </div>
      </section>

      <section className="plot-section content-section">
        <div className="plot-heading">
          <div><h2>Visual Comparison Using Matplotlib</h2></div>
        </div>
        <figure className="plot-figure">
          <img src={comparisonImage} alt="Three scatter plots compare K-Means, K-Medoids, and DBSCAN cluster assignments by normalized budget and duration." />
          <figcaption>
            The algorithms used all five normalized features to cluster projects. This plot shows
            only budget and duration for easier viewing, so the cluster colors and numbers do not match
            across methods.
          </figcaption>
        </figure>
      </section>

      <section className="takeaway-section">
        <div className="content-section takeaway-inner">
          <div><p className="section-kicker">04 / TAKEAWAY</p><h2>We found all our algorithms useful, each in different ways.<br /></h2></div>
          <div className="takeaway-copy">
            <p>K-Means separated groups best by Silhouette Score and ran fastest. DBSCAN scored best on Davies–Bouldin and uniquely surfaced isolated projects. K-Medoids offers real project examples as representatives.</p>
            <div className="disclaimer"><strong>A side note:</strong> Clusters reveal patterns in the sample that we used. On their own, they do not prove fraud, corruption, or contractor performance issues.</div>
          </div>
        </div>
      </section>

      <footer className="site-footer">
        <a className="brand footer-brand" href="#top"><span className="brand-mark">L</span><span>LEAD<span className="brand-accent">·</span>AI</span></a>
        <div className="footer-copy">
          <p>DPWH Infrastructure Transparency Dataset · BetterGov.ph</p>
          <p>Built by De Castro, Gregorio, Lanao, and Chua · CMSC 170</p>
          <p>Algorithms and metrics were implemented with Python and NumPy. Matplotlib was used for visualization. AI assistance disclosed.</p>
        </div>
        <a className="footer-link" href="https://github.com/randomchaffee/lead-ai" target="_blank" rel="noreferrer">Repository <span>↗</span></a>
      </footer>
    </main>
  )
}

export default App
