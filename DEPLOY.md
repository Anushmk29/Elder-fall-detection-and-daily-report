# Deploying ElderCare Vision

The app and a Caddy reverse proxy (automatic HTTPS from Let's Encrypt) run with Docker Compose at `https://elder.<domain>`.

## 1. VM

- **2 vCPU / 4 GB RAM / 30 GB disk** is enough for this app: one job at a time; a 52 s test clip took ~30 s end to end with 2 CPU threads (pose ~20 s, render ~9 s; LLM calls extra).
  Any Ubuntu 22.04/24.04 VPS works (Hetzner CX22/CPX21, DigitalOcean 4 GB, Lightsail 4 GB, and so on).
- To host KidsCare on the same VM, use **4 vCPU / 8 GB** and a single Caddy for both apps (see the end of this file).
- Open ports 22, 80 and 443 only.

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # log out and back in
```

## 2. DNS

Create an **A record** `elder.<domain>` pointing to the VM's IPv4 (plus AAAA for IPv6 if you have it). Wait until
`dig +short elder.<domain>` returns the IP. Caddy can only get a certificate after this resolves.

## 3. First deploy

```bash
git clone <your repo> eldercare && cd eldercare      # or: scp -r "D:\Autonoetic Edge\ElderCare" user@vm:~/eldercare
cp .env.example .env && nano .env                    # DOMAIN, OPENROUTER_API_KEY, optional DEMO_PASSCODE
cd deploy
docker compose up -d --build
docker compose logs -f                               # wait for "certificate obtained"
curl https://elder.<domain>/healthz
```

The first build takes about 5 minutes (CPU torch is about 200 MB, and the pose weights are baked in).

## 4. Updates

```bash
cd eldercare && git pull          # or copy the changed files over
cd deploy && docker compose up -d --build
docker image prune -f
```

New examples: run `scripts/process_example.py` on your PC, commit or copy `examples/<slug>/`, then rebuild as above.
Uploaded results live in the `app-data` volume and survive rebuilds until `RETENTION_HOURS` passes.

## 5. Operations

- Logs: `docker compose logs -f app`
- LLM spend today: `docker compose exec app cat /data/llm/spend.json`
- Wipe all uploads now: `docker compose exec app sh -c 'rm -rf /data/jobs/* /data/results/* /data/uploads/*'`
- Lock the demo during the summit: set `DEMO_PASSCODE` in `.env`, then run `docker compose up -d`.

## Both apps on one VM

Each app's compose file starts its own Caddy on ports 80/443, so two of them can't share a VM. Instead, run both
`app` services from one compose file in a parent folder, with one Caddyfile:

```
elder.{$DOMAIN} {
	request_body {
		max_size 110MB
	}
	reverse_proxy eldercare:8000
}
kids.{$DOMAIN} {
	request_body {
		max_size 110MB
	}
	reverse_proxy kidscare:8000
}
```

```yaml
services:
  eldercare: { build: ./ElderCare, env_file: ./ElderCare/.env, volumes: [elder-data:/data], restart: unless-stopped }
  kidscare:  { build: ./KidsCare,  env_file: ./KidsCare/.env,  volumes: [kids-data:/data],  restart: unless-stopped }
  caddy:
    image: caddy:2.10-alpine
    environment: [DOMAIN=example.com]
    ports: ["80:80", "443:443"]
    volumes: [./Caddyfile:/etc/caddy/Caddyfile:ro, caddy-data:/data, caddy-config:/config]
volumes: { elder-data: {}, kids-data: {}, caddy-data: {}, caddy-config: {} }
```
