# HELIOS-IX — WebGPU Tokamak Fusion Reactor

HELIOS-IX is an interactive 3D visualization of a magnetic confinement tokamak fusion reactor. It runs on Three.js (`0.187.0`), `WebGPURenderer`, Three Shading Language (TSL), and WGSL compute shaders. Gemini 4 Neon built this project as a browser showcase of procedural 3D engineering and GPU physics simulation.

## Features

- **WebGPU Compute Plasma** (`src/plasma/PlasmaSystem.js`): A WGSL compute shader updates 16,384 to 131,072 plasma ions per frame in GPU storage buffers. Ions follow Miller D-shaped helical magnetic flux surfaces, react to cursor movement, and strike the divertor during ELM bursts.
- **Automatic WebGL2 Fallback**: Browsers without WebGPU run an equivalent TSL fallback shader automatically. Visitors can switch backends with the `TRY WEBGL2` button or by opening `?backend=webgl`.
- **Procedural 3D Reactor** (`src/reactor/buildTokamak.js`): The scene builds 16 Toroidal Field coils, 16 vacuum vessel sectors, and 6 split-cutaway Poloidal Field rings. It also builds a central solenoid, 32 glowing tungsten divertor cassettes, and 2 Neutral Beam Injectors with rotating vacuum pumps.
- **Interactive Controls** (`src/ui/HudController.js`): Visitors can adjust cutaway and explosion (`0%` to `100%`), switch between 4 operating stages, and trigger ELM bursts or NBI pulses. They can also tune magnetic parameters, change compute particle counts, select 5 camera presets, or press `H` for Zen Mode.

## Hosting Model: Pure Static Files

This application does not require a Node.js server in production.

Running `npm run build` compiles the entire application into static files inside `dist/`:

- `dist/index.html`
- `dist/assets/index-*.js`
- `dist/assets/index-*.css`

Every 3D model, shader, compute kernel, and HUD control runs inside the visitor's browser. Because `vite.config.js` sets `base: './'`, the `dist/` folder works on a root domain, a subdomain, or a nested subpath such as `https://example.com/helios-tokamak/`.

Note: Open the built site through a web server (or HTTPS static host), not via `file://`. Browsers block ES module scripts on `file://` URLs.

## Security and Public Exposure

Publishing this project on a public domain is safe.

- **No backend attack surface**: The app contains no server code, no API routes, no database, and no file upload handlers.
- **No secrets or credentials**: The bundle stores no API keys, tokens, or internal URLs.
- **No user data collection**: The app sets no cookies, writes no `localStorage`, and sends no analytics.
- **Client-side GPU work only**: All WebGPU and WebGL computation executes on the visitor's device, not on your server.
- **One optional external request**: `index.html` loads `Inter` and `JetBrains Mono` from Google Fonts (`fonts.googleapis.com` and `fonts.gstatic.com`). If you want zero third-party requests or a strict Content Security Policy, remove the three `<link>` font tags from `index.html`. The UI then uses local system fonts.

## HTTPS Requirement for WebGPU

Web browsers expose `navigator.gpu` (WebGPU) only in secure contexts:

- `https://` on any public or private domain
- `http://localhost` or `http://127.0.0.1` for local testing

If you serve the site over plain `http://` on a domain, browsers hide WebGPU and the app falls back to `WebGL2 · TSL Fallback`. Serve the site over **HTTPS** so visitors get full WebGPU compute shaders.

## Build the Static Bundle

Requirements: Node.js 20+ (only for the build step, not on the production server).

1. Open the project folder.
2. Install dependencies:
   ```bash
   npm ci
   ```
3. Build the static bundle:
   ```bash
   npm run build
   ```
4. Upload the contents of `dist/` to your web host.

## Deployment Examples

### Option 1: Nginx (Static Directory)

Copy `dist/` to `/var/www/helios-tokamak/` and add a server block with TLS:

```nginx
server {
    listen 443 ssl http2;
    server_name helios.example.com;

    ssl_certificate /etc/letsencrypt/live/helios.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/helios.example.com/privkey.pem;

    root /var/www/helios-tokamak;
    index index.html;

    location /assets/ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Option 2: Caddy (Automatic HTTPS)

Copy `dist/` to `/srv/helios-tokamak/` and add this block to `Caddyfile`:

```caddy
helios.example.com {
    root * /srv/helios-tokamak
    encode zstd gzip
    file_server
}
```

### Option 3: Static Hosting Platforms

Upload `dist/` directly to Cloudflare Pages, Netlify, Vercel, GitHub Pages, or an S3 bucket behind CloudFront. Set the build command to `npm run build` and the output directory to `dist`.

### Option 4: Docker (Static Nginx Container)

Create a `Dockerfile` in the project root if you deploy containers:

```dockerfile
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

Place this container behind your reverse proxy with HTTPS enabled.

## Local Development

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the Vite dev server:
   ```bash
   npm run dev
   ```
3. Open `http://127.0.0.1:5179` in Chrome or Edge.
4. Preview the production bundle locally before upload:
   ```bash
   npm run build
   npm run preview
   ```

## Project Structure

```text
helios-tokamak/
├── index.html                  # App shell, HUD markup, optional font links
├── vite.config.js              # Relative base path (./) for portable static hosting
├── package.json                # Dependencies (three 0.187.0, vite)
├── dist/                       # Generated static production output (upload this)
└── src/
    ├── main.js                 # WebGPURenderer, RenderPipeline bloom, lighting, loop
    ├── style.css               # HUD styling and Zen Mode rules
    ├── reactor/
    │   ├── tokamakMath.js      # Shared Miller D-section equilibrium geometry
    │   └── buildTokamak.js     # Procedural 3D coils, vessel, divertor, NBI, hall
    ├── plasma/
    │   └── PlasmaSystem.js     # WGSL compute plasma particles, mantle, flux cage
    └── ui/
        └── HudController.js    # Controls, modes, camera presets, 3D callouts
```
