#!/usr/bin/env bash
# Sign and notarize Oncura Demo.app for Gatekeeper (Developer ID, not App Store).
#
# Prerequisites (one-time, on your Mac):
#   1. Apple Developer account
#   2. "Developer ID Application" certificate in Keychain Access
#      (developer.apple.com → Certificates → + → Developer ID Application)
#   3. App-specific password for notarytool:
#      appleid.apple.com → Sign-In and Security → App-Specific Passwords
#
# Usage (after building):
#   cd demo && python build_native.py
#   export APPLE_ID="you@example.com"
#   export APPLE_TEAM_ID="XXXXXXXXXX"
#   export APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"
#   export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
#   ./sign_mac_app.sh
#
# Or let the script pick the first Developer ID Application identity:
#   ./sign_mac_app.sh --auto-identity

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Oncura Demo"
APP_PATH="${DEMO_DIR}/dist/${APP_NAME}.app"
ENTITLEMENTS="${DEMO_DIR}/entitlements.plist"
ZIP_OUT="${DEMO_DIR}/../dist/Oncura-Demo-mac-signed.zip"
AUTO_IDENTITY=false

for arg in "$@"; do
  [[ "$arg" == "--auto-identity" ]] && AUTO_IDENTITY=true
done

if [[ ! -d "$APP_PATH" ]]; then
  echo "Missing ${APP_PATH}. Run: python build_native.py"
  exit 1
fi

if $AUTO_IDENTITY; then
  APPLE_SIGNING_IDENTITY="$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)"/\1/')"
fi

if [[ -z "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  echo "Set APPLE_SIGNING_IDENTITY or pass --auto-identity"
  echo 'Example: export APPLE_SIGNING_IDENTITY="Developer ID Application: R. Craig Stillwell (TEAMID)"'
  exit 1
fi

echo "Signing with: ${APPLE_SIGNING_IDENTITY}"
codesign --deep --force --verify --verbose=2 \
  --sign "${APPLE_SIGNING_IDENTITY}" \
  --options runtime \
  --entitlements "${ENTITLEMENTS}" \
  --timestamp \
  "${APP_PATH}"

codesign --verify --deep --strict --verbose=2 "${APP_PATH}"
spctl -a -t exec -vv "${APP_PATH}" || true

if [[ -z "${APPLE_ID:-}" || -z "${APPLE_TEAM_ID:-}" || -z "${APPLE_APP_PASSWORD:-}" ]]; then
  echo ""
  echo "Signed locally. For notarization (removes Gatekeeper warning for others), set:"
  echo "  APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD"
  echo "Then re-run this script."
  mkdir -p "$(dirname "$ZIP_OUT")"
  ditto -c -k --keepParent "${APP_PATH}" "${ZIP_OUT}"
  echo "Wrote ${ZIP_OUT} (signed, not notarized)"
  exit 0
fi

TMP_ZIP="$(mktemp -t oncura-notarize).zip"
ditto -c -k --keepParent "${APP_PATH}" "${TMP_ZIP}"

echo "Submitting to Apple notarization (may take a few minutes)..."
xcrun notarytool submit "${TMP_ZIP}" \
  --apple-id "${APPLE_ID}" \
  --team-id "${APPLE_TEAM_ID}" \
  --password "${APPLE_APP_PASSWORD}" \
  --wait

echo "Stapling ticket to app..."
xcrun stapler staple "${APP_PATH}"
spctl -a -t exec -vv "${APP_PATH}"

mkdir -p "$(dirname "$ZIP_OUT")"
ditto -c -k --keepParent "${APP_PATH}" "${ZIP_OUT}"
rm -f "${TMP_ZIP}"

echo ""
echo "Done. Upload dist/Oncura-Demo-mac-signed.zip to GitHub Releases."
echo "Users can double-click Oncura Demo.app without right-click → Open."
