# Reference Platform Model — Market-Derived Specs

Target: indoor police tactical hybrid ground-air robot.
Reference model derived from market/research platforms below. Used to calibrate
simulation dynamics, energy model, acoustic model, and mode-arbitration cost functions.

## Source platforms

| Spec | BRINC Lemur 2 (market leader) | CapsuleBot (research, closest hybrid) | TABV Fast-Lab (research, passive wheel) |
|---|---|---|---|
| Mass | 1.5 kg (3.3 lb) | 1.8 kg | 0.91 kg |
| Dimensions | 33 x 41 x 10 cm | 48 x 28 x 28 cm, wheel dia 28 cm | arm 0.23 m |
| Battery | n/a (20+ min flight) | 3000 mAh 6S = 66.6 Wh (480 g) | n/a |
| Flight power | — | 480.6 W (3.7 g/W) | — |
| Ground power | — | 5.2 W (346.2 g/W) | — |
| Noise, flight @1 m | — | 85.4 dB (84.1–87.1) | — |
| Noise, ground @1 m | — | 53.1 dB (47.4–56.7); 47.1 dB @5 m | — |
| Ambient reference | — | 42.9 dB | — |
| Speed, flight | — | ~0.5 m/s | 3 m/s max, 2.5 m/s² |
| Speed, ground | — | ~0.5 m/s | — |
| Slope / obstacle (ground) | — | ~10°, 3 cm step | — |
| Sensors | 3D LiDAR 570k pts/s, 4K + FLIR thermal, night IR | — | — |
| Compute | — | — | Jetson Xavier NX, NMPC @200 Hz |
| Comms | AES-256, mesh w/ radios | — | — |
| Special | perch, glass breaker option, 1 lb dropper, loudspeaker/mic, IP24 | actuated-wheel-rotor (4 motors total) | complementary-constraint unified NMPC |

Sources: brincdrones.com/lemur-2, arXiv:2309.09224 (CapsuleBot), arXiv:2403.00322 (TABV NMPC).

## Derived reference model ("RefBot-1")

Design point: Lemur 2 mission profile + CapsuleBot-style actuated wheels.

| Parameter | Value | Rationale |
|---|---|---|
| Total mass | 1.8 kg | Lemur 2 + wheel hardware; matches CapsuleBot measured point |
| Battery | 66.6 Wh usable (3000 mAh 6S) | CapsuleBot; yields ~20 min pure flight like Lemur 2 |
| Hover power | 300 W | Momentum-theory estimate for 1.8 kg quad w/ 7" props, FM≈0.55; CapsuleBot's 480 W is bicopter (worse disk loading) — quad layout assumed |
| Ground drive power | 8 W @ 1 m/s | CapsuleBot 5.2 W @0.5 m/s + margin, scaled linear w/ speed |
| Avionics base load | 20 W constant | LiDAR (~10 W) + Jetson-class compute (~10 W), both modes |
| Flight speed | 0–3 m/s (1.5 nominal indoor) | TABV demonstrated; Lemur-class indoor practice |
| Ground speed | 0–1.5 m/s (0.7 nominal) | CapsuleBot 0.5 measured; actuated wheels allow more |
| Noise flight @1 m | 85 dB | CapsuleBot measured |
| Noise ground @1 m | 53 dB | CapsuleBot measured |
| Ground capability | slope ≤ 10°, step ≤ 3 cm | CapsuleBot; anything above → must fly |
| Transition cost | 5 s, 15 J (land) / 10 s, 900 J (takeoff+climb 1 m) | estimate; refine from hardware tests |

## Key derived ratios (drive mode-arbitration cost function)

- Power ratio flight:ground ≈ **40:1** (320 W vs 28 W incl. avionics) → endurance: ~12 min flight-only w/ avionics vs ~2.4 h ground-only.
- Noise delta ≈ **32 dB** → flight audible through ~1–2 interior walls (TL ≈ 25–35 dB/wall); ground mode near-ambient beyond 5 m.
- Speed ratio ≈ 2:1 flight advantage → flying saves time but costs ~80x energy per meter.

## Open items (measure on real hardware, step 3 of dev plan)

- [ ] Actual hover power vs speed curve (P(v))
- [ ] Ground power vs speed, per floor surface (carpet/tile/concrete)
- [ ] Transition energy + duration (land, takeoff)
- [ ] Noise spectrum per mode, wall transmission loss in target buildings
- [ ] Battery sag / usable-capacity fraction under 300 W draw
