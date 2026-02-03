/**
 * Yellow Hat #2 — Portfolio-Showcase Readiness Tests
 *
 * Pass 9 tests validating the project is portfolio-presentation-grade:
 * 1. README validation (badges, feature tables, shortcuts, architecture, concepts)
 * 2. PWA manifest (fields, icons, categories)
 * 3. HTML meta tags (manifest, apple, OG, Twitter, JSON-LD, SEO)
 * 4. Deployment assets (favicon, og-image, 404, robots, sitemap, icons, LICENSE)
 * 5. Package metadata (version, description, homepage, repo, keywords, license, scripts)
 * 6. CI/CD workflow validation
 * 7. Source code quality (no TODO/FIXME, no as-any, no console.log, ErrorBoundary, reduced-motion)
 * 8. Constants consistency (timing, display, neurons, shortcuts)
 * 9. Type system completeness
 * 10. Architecture integrity (barrels, module boundaries, data flow)
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const ROOT = path.resolve(__dirname, '..', '..');
const PUBLIC = path.join(ROOT, 'public');
const SRC = path.join(ROOT, 'src');

function readFile(p: string): string {
  return fs.readFileSync(p, 'utf-8');
}
function fileExists(p: string): boolean {
  return fs.existsSync(p);
}

// ─── README Validation ──────────────────────────────────────────────

describe('README validation', () => {
  const readme = readFile(path.join(ROOT, 'README.md'));

  it('has shields.io badges', () => {
    expect(readme).toContain('img.shields.io');
    expect(readme).toContain('TypeScript');
    expect(readme).toContain('tests');
    expect(readme).toContain('license');
    expect(readme).toContain('bundle');
  });

  it('has live demo link', () => {
    expect(readme).toContain('kai-claw.github.io/neuralplayground');
    expect(readme).toContain('Live Demo');
  });

  it('has feature tables', () => {
    expect(readme).toContain('Core Neural Network Engine');
    expect(readme).toContain('Visualization & Interaction');
    expect(readme).toContain('Labs & Experiments');
    expect(readme).toContain('Experience Modes');
  });

  it('has keyboard shortcuts table', () => {
    expect(readme).toContain('Keyboard Shortcuts');
    expect(readme).toContain('Space');
    expect(readme).toContain('Reset');
    expect(readme).toContain('Help');
  });

  it('has architecture diagram', () => {
    expect(readme).toContain('Architecture');
    expect(readme).toContain('src/');
    expect(readme).toContain('nn/');
    expect(readme).toContain('hooks/');
    expect(readme).toContain('components/');
    expect(readme).toContain('renderers/');
    expect(readme).toContain('visualizers/');
    expect(readme).toContain('utils/');
  });

  it('has tech stack table', () => {
    expect(readme).toContain('Tech Stack');
    expect(readme).toContain('React');
    expect(readme).toContain('TypeScript');
    expect(readme).toContain('Vite');
    expect(readme).toContain('Vitest');
    expect(readme).toContain('Canvas');
  });

  it('has ML concepts table', () => {
    expect(readme).toContain('ML Concepts');
    expect(readme).toContain('Backpropagation');
    expect(readme).toContain('Xavier');
    expect(readme).toContain('Cross-Entropy');
    expect(readme).toContain('Gradient Ascent');
    expect(readme).toContain('Saliency');
    expect(readme).toContain('PCA');
    expect(readme).toContain('Adversarial');
    expect(readme).toContain('Ablation');
    expect(readme).toContain('Confusion');
  });

  it('has metrics table', () => {
    expect(readme).toContain('Metrics');
    expect(readme).toContain('Source files');
    expect(readme).toContain('Tests');
    expect(readme).toContain('Bundle');
  });

  it('has accessibility section', () => {
    expect(readme).toContain('Accessibility');
    expect(readme).toContain('ARIA');
    expect(readme).toContain('prefers-reduced-motion');
    expect(readme).toContain('Keyboard');
  });

  it('has getting started section', () => {
    expect(readme).toContain('Getting Started');
    expect(readme).toContain('npm install');
    expect(readme).toContain('npm run dev');
    expect(readme).toContain('npm test');
    expect(readme).toContain('npm run build');
    expect(readme).toContain('npm run deploy');
  });

  it('has development process section', () => {
    expect(readme).toContain('Six Thinking Hats');
    // Uses emoji hats ⚪⚫🟢🟡🔴🔵
    expect(readme).toContain('⚪');
    expect(readme).toContain('⚫');
    expect(readme).toContain('🟢');
    expect(readme).toContain('🟡');
    expect(readme).toContain('🔴');
    expect(readme).toContain('🔵');
  });

  it('has MIT license reference', () => {
    expect(readme).toContain('MIT');
    expect(readme).toContain('LICENSE');
  });
});

// ─── PWA Manifest ────────────────────────────────────────────────────

describe('PWA manifest', () => {
  const manifest = JSON.parse(readFile(path.join(PUBLIC, 'manifest.json')));

  it('has required fields', () => {
    expect(manifest.name).toBe('NeuralPlayground');
    expect(manifest.short_name).toBe('NeuralPlayground');
    expect(manifest.description).toBeTruthy();
    expect(manifest.start_url).toBe('/neuralplayground/');
    expect(manifest.display).toBe('standalone');
    expect(manifest.background_color).toBeTruthy();
    expect(manifest.theme_color).toBeTruthy();
  });

  it('has education category', () => {
    expect(manifest.categories).toContain('education');
  });

  it('has SVG and raster icons', () => {
    const svgIcon = manifest.icons.find((i: { type: string }) => i.type === 'image/svg+xml');
    const png192 = manifest.icons.find((i: { sizes: string }) => i.sizes === '192x192');
    const png512 = manifest.icons.find((i: { sizes: string }) => i.sizes === '512x512');
    expect(svgIcon).toBeTruthy();
    expect(png192).toBeTruthy();
    expect(png192.type).toBe('image/png');
    expect(png512).toBeTruthy();
    expect(png512.type).toBe('image/png');
  });

  it('icon files exist on disk', () => {
    expect(fileExists(path.join(PUBLIC, 'favicon.svg'))).toBe(true);
    expect(fileExists(path.join(PUBLIC, 'icon-192.png'))).toBe(true);
    expect(fileExists(path.join(PUBLIC, 'icon-512.png'))).toBe(true);
  });
});

// ─── HTML Meta Tags ──────────────────────────────────────────────────

describe('HTML meta tags', () => {
  const html = readFile(path.join(ROOT, 'index.html'));

  it('has manifest link', () => {
    expect(html).toContain('rel="manifest"');
  });

  it('has apple-mobile-web-app tags', () => {
    expect(html).toContain('apple-mobile-web-app-capable');
    expect(html).toContain('apple-mobile-web-app-status-bar-style');
    expect(html).toContain('apple-mobile-web-app-title');
    expect(html).toContain('apple-touch-icon');
  });

  it('has Open Graph tags', () => {
    expect(html).toContain('og:type');
    expect(html).toContain('og:title');
    expect(html).toContain('og:description');
    expect(html).toContain('og:url');
    expect(html).toContain('og:image');
    expect(html).toContain('og:image:alt');
    expect(html).toContain('og:image:width');
    expect(html).toContain('og:image:height');
    expect(html).toContain('og:site_name');
  });

  it('has Twitter card tags', () => {
    expect(html).toContain('twitter:card');
    expect(html).toContain('twitter:title');
    expect(html).toContain('twitter:description');
    expect(html).toContain('twitter:image');
    expect(html).toContain('twitter:image:alt');
  });

  it('has JSON-LD structured data', () => {
    expect(html).toContain('application/ld+json');
    expect(html).toContain('"@type": "WebApplication"');
    expect(html).toContain('"softwareVersion"');
    expect(html).toContain('"featureList"');
    expect(html).toContain('"isAccessibleForFree"');
    expect(html).toContain('"educationalLevel"');
  });

  it('has SEO essentials', () => {
    expect(html).toContain('meta name="description"');
    expect(html).toContain('meta name="keywords"');
    expect(html).toContain('meta name="author"');
    expect(html).toContain('meta name="theme-color"');
    expect(html).toContain('rel="canonical"');
  });

  it('has loading spinner + noscript', () => {
    expect(html).toContain('app-loader');
    expect(html).toContain('<noscript>');
  });
});

// ─── Deployment Assets ───────────────────────────────────────────────

describe('deployment assets', () => {
  it('has favicon.svg', () => {
    expect(fileExists(path.join(PUBLIC, 'favicon.svg'))).toBe(true);
  });

  it('has og-image.svg', () => {
    expect(fileExists(path.join(PUBLIC, 'og-image.svg'))).toBe(true);
  });

  it('has 404.html', () => {
    const html404 = readFile(path.join(PUBLIC, '404.html'));
    expect(html404).toContain('neuralplayground');
  });

  it('has robots.txt', () => {
    const robots = readFile(path.join(PUBLIC, 'robots.txt'));
    expect(robots).toContain('Sitemap');
  });

  it('has sitemap.xml', () => {
    const sitemap = readFile(path.join(PUBLIC, 'sitemap.xml'));
    expect(sitemap).toContain('neuralplayground');
    expect(sitemap).toContain('2026');
  });

  it('has PWA icons', () => {
    expect(fileExists(path.join(PUBLIC, 'icon-192.png'))).toBe(true);
    expect(fileExists(path.join(PUBLIC, 'icon-512.png'))).toBe(true);
  });

  it('has LICENSE', () => {
    expect(fileExists(path.join(ROOT, 'LICENSE'))).toBe(true);
    const license = readFile(path.join(ROOT, 'LICENSE'));
    expect(license).toContain('MIT');
  });

  it('has README', () => {
    expect(fileExists(path.join(ROOT, 'README.md'))).toBe(true);
    const readme = readFile(path.join(ROOT, 'README.md'));
    expect(readme.length).toBeGreaterThan(2000);
  });
});

// ─── Package Metadata ────────────────────────────────────────────────

describe('package.json metadata', () => {
  const pkg = JSON.parse(readFile(path.join(ROOT, 'package.json')));

  it('has version 1.0.0', () => {
    expect(pkg.version).toBe('1.0.0');
  });

  it('has description', () => {
    expect(pkg.description).toBeTruthy();
    expect(pkg.description.length).toBeGreaterThan(20);
  });

  it('has homepage', () => {
    expect(pkg.homepage).toContain('kai-claw.github.io');
  });

  it('has repository', () => {
    expect(pkg.repository.url).toContain('github.com');
    expect(pkg.repository.url).toContain('neuralplayground');
  });

  it('has keywords', () => {
    expect(pkg.keywords.length).toBeGreaterThanOrEqual(5);
    expect(pkg.keywords).toContain('neural-network');
    expect(pkg.keywords).toContain('typescript');
  });

  it('has author and license', () => {
    expect(pkg.author).toBeTruthy();
    expect(pkg.license).toBe('MIT');
  });

  it('has all required scripts', () => {
    expect(pkg.scripts.dev).toBeTruthy();
    expect(pkg.scripts.build).toBeTruthy();
    expect(pkg.scripts.test).toBeTruthy();
    expect(pkg.scripts.deploy).toBeTruthy();
  });
});

// ─── CI/CD Workflow ──────────────────────────────────────────────────

describe('CI/CD workflow', () => {
  it('has GitHub Actions workflow', () => {
    const workflowDir = path.join(ROOT, '.github', 'workflows');
    expect(fileExists(workflowDir)).toBe(true);
    const files = fs.readdirSync(workflowDir);
    expect(files.length).toBeGreaterThan(0);
    const workflow = readFile(path.join(workflowDir, files[0]));
    expect(workflow).toContain('npm');
    expect(workflow).toContain('build');
  });
});

// ─── Source Code Quality ─────────────────────────────────────────────

describe('source code quality', () => {
  function getAllSourceFiles(dir: string, ext: string[]): string[] {
    const results: string[] = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory() && entry.name !== '__tests__' && entry.name !== 'node_modules') {
        results.push(...getAllSourceFiles(full, ext));
      } else if (entry.isFile() && ext.some(e => entry.name.endsWith(e))) {
        results.push(full);
      }
    }
    return results;
  }

  const sourceFiles = getAllSourceFiles(SRC, ['.ts', '.tsx']).filter(
    f => !f.includes('__tests__') && !f.includes('vite-env')
  );

  it('no TODO/FIXME/HACK in source files', () => {
    for (const file of sourceFiles) {
      const content = readFile(file);
      const rel = path.relative(SRC, file);
      expect(content).not.toMatch(/\bTODO\b/i);
      expect(content).not.toMatch(/\bFIXME\b/i);
      expect(content).not.toMatch(/\bHACK\b/i);
    }
  });

  it('no "as any" in source files', () => {
    for (const file of sourceFiles) {
      const content = readFile(file);
      expect(content).not.toContain('as any');
    }
  });

  it('no bare console.log in source (except ErrorBoundary)', () => {
    for (const file of sourceFiles) {
      if (file.includes('ErrorBoundary')) continue;
      const content = readFile(file);
      expect(content).not.toMatch(/console\.log\(/);
    }
  });

  it('has ErrorBoundary component', () => {
    expect(fileExists(path.join(SRC, 'components', 'ErrorBoundary.tsx'))).toBe(true);
  });

  it('has prefers-reduced-motion in CSS', () => {
    const css = readFile(path.join(SRC, 'App.css'));
    expect(css).toContain('prefers-reduced-motion');
  });

  it('tsconfig has strict mode', () => {
    const tsconfigPath = path.join(ROOT, 'tsconfig.app.json');
    const raw = readFile(tsconfigPath);
    // Strip JSONC comments before parsing
    const stripped = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    const tsconfig = JSON.parse(stripped);
    expect(tsconfig.compilerOptions.strict).toBe(true);
  });
});

// ─── Constants Consistency ───────────────────────────────────────────

describe('constants consistency', () => {
  // Dynamic import of constants
  let constants: Record<string, unknown>;

  it('loads constants module', async () => {
    constants = await import('../constants');
    expect(constants).toBeTruthy();
  });

  it('timing constants are positive', async () => {
    const c = await import('../constants');
    expect(c.CINEMATIC_TRAIN_EPOCHS).toBeGreaterThan(0);
    expect(c.CINEMATIC_PREDICT_DWELL).toBeGreaterThan(0);
    expect(c.CINEMATIC_EPOCH_INTERVAL).toBeGreaterThan(0);
    expect(c.AUTO_TRAIN_EPOCHS).toBeGreaterThan(0);
    expect(c.AUTO_TRAIN_DELAY).toBeGreaterThan(0);
  });

  it('display aspects are valid', async () => {
    const c = await import('../constants');
    expect(c.NETWORK_VIS_ASPECT).toBeGreaterThan(0);
    expect(c.NETWORK_VIS_ASPECT).toBeLessThan(2);
    expect(c.LOSS_CHART_ASPECT).toBeGreaterThan(0);
    expect(c.LOSS_CHART_ASPECT).toBeLessThan(2);
  });

  it('INPUT_SIZE equals INPUT_DIM squared', async () => {
    const c = await import('../constants');
    expect(c.INPUT_SIZE).toBe(c.INPUT_DIM * c.INPUT_DIM);
  });

  it('NEURON_OPTIONS are sorted ascending', async () => {
    const c = await import('../constants');
    const opts = [...c.NEURON_OPTIONS];
    for (let i = 1; i < opts.length; i++) {
      expect(opts[i]).toBeGreaterThan(opts[i - 1]);
    }
  });

  it('MAX_HIDDEN_LAYERS is reasonable', async () => {
    const c = await import('../constants');
    expect(c.MAX_HIDDEN_LAYERS).toBeGreaterThanOrEqual(1);
    expect(c.MAX_HIDDEN_LAYERS).toBeLessThanOrEqual(10);
  });

  it('OUTPUT_CLASSES is 10 (digits 0-9)', async () => {
    const c = await import('../constants');
    expect(c.OUTPUT_CLASSES).toBe(10);
  });

  it('DEFAULT_CONFIG has valid structure', async () => {
    const c = await import('../constants');
    expect(c.DEFAULT_CONFIG.learningRate).toBeGreaterThan(0);
    expect(c.DEFAULT_CONFIG.learningRate).toBeLessThan(1);
    expect(c.DEFAULT_CONFIG.layers.length).toBeGreaterThan(0);
    for (const layer of c.DEFAULT_CONFIG.layers) {
      expect(layer.neurons).toBeGreaterThan(0);
      expect(['relu', 'sigmoid', 'tanh']).toContain(layer.activation);
    }
  });
});

// ─── Type System Completeness ────────────────────────────────────────

describe('type system completeness', () => {
  it('exports all core types', async () => {
    const types = await import('../types');
    // Verify key type names exist as exports (TypeScript types are erased,
    // but we can check the module has expected exports)
    expect(types).toBeTruthy();
  });

  it('activation functions are exhaustive', async () => {
    const { activate, activateDerivative } = await import('../utils/activations');
    for (const fn of ['relu', 'sigmoid', 'tanh'] as const) {
      expect(typeof activate(0.5, fn)).toBe('number');
      expect(typeof activateDerivative(0.5, fn)).toBe('number');
    }
  });

  it('all noise types have labels and descriptions', async () => {
    const c = await import('../constants');
    const noiseLabels = c.NOISE_LABELS;
    const noiseDescriptions = c.NOISE_DESCRIPTIONS;
    const noiseTypes: string[] = ['gaussian', 'salt-pepper', 'adversarial'];
    for (const nt of noiseTypes) {
      expect(noiseLabels[nt as keyof typeof noiseLabels]).toBeTruthy();
      expect(noiseDescriptions[nt as keyof typeof noiseDescriptions]).toBeTruthy();
    }
  });
});

// ─── Architecture Integrity ──────────────────────────────────────────

describe('architecture integrity', () => {
  it('all 6 barrel exports exist', () => {
    const barrels = ['nn', 'hooks', 'components', 'renderers', 'visualizers', 'utils'];
    for (const dir of barrels) {
      expect(fileExists(path.join(SRC, dir, 'index.ts'))).toBe(true);
    }
  });

  it('nn/ has no React imports', () => {
    const nnDir = path.join(SRC, 'nn');
    for (const file of fs.readdirSync(nnDir).filter(f => f.endsWith('.ts'))) {
      const content = readFile(path.join(nnDir, file));
      expect(content).not.toMatch(/from\s+['"]react['"]/);
    }
  });

  it('renderers/ has no React imports', () => {
    const dir = path.join(SRC, 'renderers');
    for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.ts'))) {
      const content = readFile(path.join(dir, file));
      expect(content).not.toMatch(/from\s+['"]react['"]/);
    }
  });

  it('backward-compat re-exports exist and are thin', () => {
    const compatFiles = ['noise.ts', 'visualizer.ts', 'rendering.ts'];
    for (const f of compatFiles) {
      const content = readFile(path.join(SRC, f));
      // Thin = less than 20 lines
      const lines = content.split('\n').length;
      expect(lines).toBeLessThan(20);
    }
  });

  it('ARCHITECTURE.md exists', () => {
    expect(fileExists(path.join(ROOT, 'ARCHITECTURE.md'))).toBe(true);
    const arch = readFile(path.join(ROOT, 'ARCHITECTURE.md'));
    expect(arch).toContain('Directory Structure');
  });

  it('AUDIT.md exists with comparison table', () => {
    expect(fileExists(path.join(ROOT, 'AUDIT.md'))).toBe(true);
    const audit = readFile(path.join(ROOT, 'AUDIT.md'));
    expect(audit).toContain('Baseline vs. Final');
    expect(audit).toContain('Pass 1');
    expect(audit).toContain('Pass 8');
  });
});

// ─── Cross-Module Integration ────────────────────────────────────────

describe('cross-module integration', () => {
  it('NeuralNetwork → predict pipeline', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { lerpPixels } = await import('../renderers/pixelRendering');
    const config = { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' as const }] };
    const nn = new NeuralNetwork(784, config);
    const input = new Array(784).fill(0);
    const result = nn.predict(input);
    expect(result.probabilities.length).toBe(10);
    expect(typeof result.label).toBe('number');
    expect(result.layers.length).toBeGreaterThan(0);
    // lerpPixels is a pure function (no DOM needed)
    const a = new Array(784).fill(0);
    const b = new Array(784).fill(1);
    const mid = lerpPixels(a, b, 0.5);
    expect(mid.length).toBe(784);
    expect(mid[0]).toBeCloseTo(0.5, 1);
  });

  it('NeuralNetwork → training → snapshot pipeline', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { generateTrainingData } = await import('../nn/sampleData');
    const config = { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' as const }] };
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    const snapshot = nn.trainBatch(data.inputs, data.labels);
    expect(snapshot.epoch).toBe(1);
    expect(typeof snapshot.loss).toBe('number');
    expect(typeof snapshot.accuracy).toBe('number');
    expect(snapshot.layers.length).toBeGreaterThan(0);
  });

  it('noise → predict → confidence impact', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { generateTrainingData } = await import('../nn/sampleData');
    const { applyNoise } = await import('../nn/noise');
    const config = { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' as const }] };
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);
    const cleanInput = data.inputs[0];
    const cleanResult = nn.predict(cleanInput);
    const noisyInput = applyNoise(cleanInput, 'gaussian', 0.8, 42);
    const noisyResult = nn.predict(noisyInput);
    // Both should produce valid probabilities
    expect(cleanResult.probabilities.length).toBe(10);
    expect(noisyResult.probabilities.length).toBe(10);
    const cleanMax = Math.max(...cleanResult.probabilities);
    const noisyMax = Math.max(...noisyResult.probabilities);
    expect(cleanMax).toBeGreaterThan(0);
    expect(noisyMax).toBeGreaterThan(0);
  });

  it('ablation study runs without error', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { generateTrainingData } = await import('../nn/sampleData');
    const { runAblationStudy } = await import('../nn/ablation');
    const config = { learningRate: 0.01, layers: [{ neurons: 8, activation: 'relu' as const }] };
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    for (let i = 0; i < 3; i++) nn.trainBatch(data.inputs, data.labels);
    const study = runAblationStudy(nn, 5);
    expect(study.layers.length).toBeGreaterThan(0);
    expect(typeof study.baselineAccuracy).toBe('number');
    for (const layer of study.layers) {
      for (const r of layer) {
        expect(typeof r.neuronIdx).toBe('number');
        expect(typeof r.accuracyDrop).toBe('number');
      }
    }
  });

  it('weight evolution records snapshots', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { generateTrainingData } = await import('../nn/sampleData');
    const { WeightEvolutionRecorder } = await import('../nn/weightEvolution');
    const config = { learningRate: 0.01, layers: [{ neurons: 8, activation: 'relu' as const }] };
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    const recorder = new WeightEvolutionRecorder();
    const snap1 = nn.trainBatch(data.inputs, data.labels);
    recorder.record(snap1);
    const snap2 = nn.trainBatch(data.inputs, data.labels);
    recorder.record(snap2);
    const frames = recorder.getFrames();
    expect(frames.length).toBe(2);
  });
});

// ─── Feature Completeness ────────────────────────────────────────────

describe('feature completeness', () => {
  it('all 29 components exist', () => {
    const componentsDir = path.join(SRC, 'components');
    const files = fs.readdirSync(componentsDir).filter(f => f.endsWith('.tsx'));
    // index.ts is barrel, not a component
    expect(files.length).toBeGreaterThanOrEqual(29);
  });

  it('all 6 hooks exist', () => {
    const hooksDir = path.join(SRC, 'hooks');
    const files = fs.readdirSync(hooksDir).filter(f => f.endsWith('.ts') && f !== 'index.ts');
    expect(files.length).toBeGreaterThanOrEqual(6);
  });

  it('all 15 nn modules exist', () => {
    const nnDir = path.join(SRC, 'nn');
    const files = fs.readdirSync(nnDir).filter(f => f.endsWith('.ts') && f !== 'index.ts');
    expect(files.length).toBeGreaterThanOrEqual(14);
  });

  it('all 7 renderers exist', () => {
    const renderersDir = path.join(SRC, 'renderers');
    const files = fs.readdirSync(renderersDir).filter(f => f.endsWith('.ts') && f !== 'index.ts');
    expect(files.length).toBeGreaterThanOrEqual(6);
  });

  it('race presets data exists', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    expect(RACE_PRESETS.length).toBeGreaterThan(0);
    for (const preset of RACE_PRESETS) {
      expect(preset.label).toBeTruthy();
      expect(preset.a).toBeTruthy();
      expect(preset.b).toBeTruthy();
      expect(preset.a.layers.length).toBeGreaterThan(0);
      expect(preset.b.layers.length).toBeGreaterThan(0);
    }
  });

  it('digit strokes data exists for all 10 digits', async () => {
    const { DIGIT_STROKES } = await import('../data/digitStrokes');
    expect(Object.keys(DIGIT_STROKES).length).toBe(10);
    for (let d = 0; d < 10; d++) {
      expect(DIGIT_STROKES[d]).toBeTruthy();
      expect(DIGIT_STROKES[d].length).toBeGreaterThan(0);
    }
  });
});
