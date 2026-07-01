# Signing the Mac demo (Developer ID)

Use this when you want **double-click launch without “unidentified developer”** — still **not** App Store distribution and **not** required for research demos.

## What you need (you likely have most of this)

| Item | Where |
|------|--------|
| Apple Developer Program | [developer.apple.com](https://developer.apple.com) ($99/yr — you said you have this) |
| **Developer ID Application** certificate | Certificates → **+** → **Developer ID Application** → install in Keychain |
| App-specific password | [appleid.apple.com](https://appleid.apple.com) → App-Specific Passwords |
| Team ID | Membership details in Developer portal |

**Do not** use “Apple Distribution” (App Store) or “Mac Development” for public download — use **Developer ID Application**.

## Local sign + notarize (recommended)

```bash
cd demo
python3 -m venv .build-venv && .build-venv/bin/pip install -r requirements_build.txt
.build-venv/bin/python build_native.py

export APPLE_ID="your@email.com"
export APPLE_TEAM_ID="XXXXXXXXXX"
export APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"
./sign_mac_app.sh --auto-identity
```

Output: `dist/Oncura-Demo-mac-signed.zip` with a **stapled, notarized** `Oncura Demo.app`.

Upload that ZIP to the [demo release](https://github.com/rstil2/cancer-alpha/releases/tag/demo) (replace `Oncura-Demo-mac.zip`).

## What this gives you

| Stage | User experience |
|-------|-----------------|
| Unsigned (current CI build) | First open: right-click → **Open** once |
| Signed only | Better, but Gatekeeper may still warn |
| **Signed + notarized + stapled** | Normal double-click for other Macs |

## Optional: CI signing via GitHub Actions

To automate on push, add repository **Secrets**:

| Secret | Value |
|--------|--------|
| `APPLE_CERTIFICATE_P12_BASE64` | Base64 of exported `.p12` (Keychain → cert → Export) |
| `APPLE_CERTIFICATE_PASSWORD` | `.p12` export password |
| `APPLE_ID` | Apple ID email |
| `APPLE_TEAM_ID` | 10-character team ID |
| `APPLE_APP_PASSWORD` | App-specific password |

Then enable the signing step in `.github/workflows/demo-native.yml` (see commented block in workflow).

**Security:** treat the `.p12` like a password; rotate if leaked.

## App Store / paid distribution

That is a **different path** (not needed for interview or research):

- Mac App Store → “Apple Distribution” cert, sandboxing, review, revenue share
- Direct sales → Developer ID + notarization (what this doc covers) + your own license/payment page

For Oncura today: **Developer ID + notarize** the demo ZIP is the right level.

## Windows note

Avoiding SmartScreen warnings needs an **Authenticode** cert (e.g. DigiCert, Sectigo) — separate purchase. Optional later; users click “More info → Run anyway” for now.

## Troubleshooting

```bash
# List signing identities
security find-identity -v -p codesigning

# Verify app signature
codesign --verify --deep --strict -vv "dist/Oncura Demo.app"

# Check Gatekeeper assessment
spctl -a -t exec -vv "dist/Oncura Demo.app"
```

If notarization fails, open the log URL from `notarytool` output — PyInstaller apps usually need `demo/entitlements.plist` (already in repo).
