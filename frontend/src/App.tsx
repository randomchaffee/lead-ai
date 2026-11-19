import './App.css'

const methods = [
  {
    name: 'K-Means',
    type: 'Centroid-Based',
    description:
      'Partitioned the projects into four clusters by assigning each project to the nearest centroid and updating each centroid as the mean of its members.',
  },
  {
    name: 'K-Medoids',
    type: 'Medoid-Based',
    description:
      'Used PAM to select representative projects as medoids, build an initial solution, and improve it through swaps.',
  },
  {
    name: 'DBSCAN',
    type: 'Density-Based',
    description:
      'Found dense regions using eps = 0.15 and minPts = 5. Projects outside dense regions were labeled as noise.',
  },
]

const features = [
  ['Budget', 'Raw contract cost in Philippine pesos'],
  ['Duration', 'Calendar days between start and completion'],
  ['Contractor Count', 'Number of projects handled by the contractor'],
  ['Average Contractor Budget', 'Mean budget across the contractor projects'],
  ['Contractor Delay Rate', 'Terminated or missing-completion projects per contractor'],
]

const results = [
  ['K-Means', '4', 'N/A', '0.3888', '1.0020', '0.0124 s'],
  ['K-Medoids', '4', 'N/A', '0.2621', '1.2323', '30.9019 s'],
  ['DBSCAN', '4', '47', '0.3601', '0.7818', '9.2070 s'],
]

function App() {
  const comparisonImage = `${import.meta.env.BASE_URL}methods_comparison.png`

  return (
    <main>
      <nav className="nav">
        <strong>LEAD-AI</strong>
        <div>
          <a href="#overview">Overview</a>
          <a href="#dataset">Dataset</a>
          <a href="#methods">Methods</a>
          <a href="#results">Results</a>
        </div>
      </nav>

      <header className="hero" id="overview">
        <p className="eyebrow">CMSC 170 AI Project</p>
        <h1>LEAD-AI</h1>
        <p className="subtitle">
          Leading Evaluation and Anomaly Detection for Philippine Public Works
        </p>
        <p className="group">
          De Castro, Amiel Vincent · Gregorio, Evan · Lanao, Lardner · Chua, Denise Kirssy
        </p>
      </header>

      <section>
        <p className="section-number">01</p>
        <h2>Problem and Motivation</h2>
        <p>
          Public infrastructure projects can experience cost, duration, and contractor
          performance risks. LEAD-AI applies unsupervised clustering to DPWH contract
          records to discover project patterns and isolate potentially anomalous projects
          for further audit review.
        </p>
        <p className="callout">
          This system identifies statistical patterns only. It does not prove fraud,
          corruption, or misconduct.
        </p>
      </section>

      <section id="dataset">
        <p className="section-number">02</p>
        <h2>Dataset and Preprocessing</h2>
        <p>
          The DPWH Infrastructure Transparency Dataset was obtained from the BetterGov.ph
          Open Data Portal and contains 248,220 source records.
        </p>
        <div className="stats">
          <div><strong>2021–2023</strong><span>Contract start years</span></div>
          <div><strong>3+</strong><span>Projects per contractor</span></div>
          <div><strong>1,600</strong><span>Randomly sampled projects</span></div>
          <div><strong>Seed 42</strong><span>Reproducible sampling</span></div>
        </div>
        <div className="feature-grid">
          {features.map(([name, description]) => (
            <article key={name}>
              <h3>{name}</h3>
              <p>{description}</p>
            </article>
          ))}
        </div>
        <p>
          All five features were manually Min-Max normalized to the range [0.0, 1.0].
          Contractor statistics were computed from the 1,600 sampled projects. Euclidean
          distance was used consistently across all methods.
        </p>
      </section>

      <section id="methods">
        <p className="section-number">03</p>
        <h2>Methods</h2>
        <div className="method-grid">
          {methods.map((method) => (
            <article className="card" key={method.name}>
              <span>{method.type}</span>
              <h3>{method.name}</h3>
              <p>{method.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section>
        <p className="section-number">04</p>
        <h2>Experimental Setup</h2>
        <div className="two-column">
          <div>
            <h3>Configuration</h3>
            <ul>
              <li>K-Means: k = 4, 300 iterations maximum</li>
              <li>K-Medoids: k = 4 using PAM BUILD and SWAP</li>
              <li>DBSCAN: eps = 0.15, minPts = 5</li>
              <li>Random seed: 42</li>
            </ul>
          </div>
          <div>
            <h3>Evaluation Metrics</h3>
            <ul>
              <li>Silhouette Score: higher is better</li>
              <li>Davies-Bouldin Index: lower is better</li>
              <li>Runtime: secondary execution comparison</li>
              <li>DBSCAN metrics exclude noise points labeled -1</li>
            </ul>
          </div>
        </div>
      </section>

      <section id="results">
        <p className="section-number">05</p>
        <h2>Results</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Clusters</th>
                <th>Noise</th>
                <th>Silhouette ↑</th>
                <th>D-B Index ↓</th>
                <th>Runtime</th>
              </tr>
            </thead>
            <tbody>
              {results.map((row) => (
                <tr key={row[0]}>
                  <td><strong>{row[0]}</strong></td>
                  {row.slice(1).map((value) => <td key={value}>{value}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <figure>
          <img src={comparisonImage} alt="Comparison of K-Means, K-Medoids, and DBSCAN clusters" />
          <figcaption>
            Two-dimensional projection of budget versus duration. All algorithms used all five
            normalized features; this plot shows only two dimensions for interpretation.
          </figcaption>
        </figure>
      </section>

      <section>
        <p className="section-number">06</p>
        <h2>Discussion</h2>
        <ul className="discussion">
          <li><strong>K-Means</strong> had the highest Silhouette Score and was the fastest method.</li>
          <li><strong>K-Medoids</strong> produced representative real projects, but had the lowest Silhouette Score and longest runtime.</li>
          <li><strong>DBSCAN</strong> had the lowest Davies-Bouldin Index and identified 47 isolated projects as noise.</li>
          <li>The metrics disagree on the overall winner, so the final choice depends on the desired behavior: fixed segmentation, representative projects, or anomaly detection.</li>
        </ul>
        <h3>Limitations and Improvements</h3>
        <p>
          The visualizations use only two feature dimensions. Contractor profiles were derived
          from the sampled subset rather than the complete filtered dataset. Missing completion
          dates and terminated statuses were used as a delay proxy. Future work may tune DBSCAN
          parameters, test additional feature combinations, or validate clusters with domain experts.
        </p>
      </section>

      <section>
        <p className="section-number">07</p>
        <h2>Conclusion</h2>
        <p>
          No method is uniformly best across both metrics. K-Means was strongest by Silhouette
          Score and runtime, while DBSCAN was strongest by Davies-Bouldin and unique in
          identifying anomalous projects. K-Medoids remains useful for interpretable medoid
          projects. K-Means or DBSCAN is the preferred choice depending on whether the goal is
          efficient fixed partitioning or density-based anomaly screening.
        </p>
      </section>

      <footer>
        <div>
          <h2>References and Disclosure</h2>
          <p>
            Data source: BetterGov.ph DPWH Infrastructure Transparency Dataset.
            All preprocessing, clustering, and metrics were implemented from scratch with
            NumPy and Python standard libraries. Matplotlib was used for visualization.
          </p>
          <p>
            AI assistants were used to help prepare documentation and debug implementation.
            The group reviewed and verified every submitted result and line of implementation.
          </p>
        </div>
        <div>
          <h3>Repository</h3>
          <a href="https://github.com/randomchaffee/lead-ai" target="_blank" rel="noreferrer">
            github.com/randomchaffee/lead-ai
          </a>
        </div>
      </footer>
    </main>
  )
}

export default App
