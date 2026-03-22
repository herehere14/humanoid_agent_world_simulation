/** Post-processing pipeline — bloom, SSAO, tone mapping, vignette, color grading.
 *
 *  This is what separates "dev demo" from "Fortnite-tier".
 */

import {
  EffectComposer,
  Bloom,
  Vignette,
  SMAA,
  ToneMapping,
  N8AO,
  BrightnessContrast,
  HueSaturation,
} from '@react-three/postprocessing';
import { ToneMappingMode } from 'postprocessing';

export function PostProcessing() {
  return (
    <EffectComposer multisampling={0}>
      {/* Screen-space ambient occlusion — soft contact shadows everywhere */}
      <N8AO
        aoRadius={0.8}
        intensity={2.5}
        distanceFalloff={0.6}
        quality="medium"
      />

      {/* Bloom — glowing windows, lamps, emotional auras, interaction sparks */}
      <Bloom
        intensity={0.4}
        luminanceThreshold={0.6}
        luminanceSmoothing={0.4}
        mipmapBlur
        radius={0.7}
      />

      {/* Color grading — punchy saturated Fortnite look */}
      <BrightnessContrast
        brightness={0.02}
        contrast={0.12}
      />
      <HueSaturation
        saturation={0.15}
      />

      {/* Filmic tone mapping */}
      <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />

      {/* Subtle vignette */}
      <Vignette
        offset={0.35}
        darkness={0.4}
      />

      {/* Anti-aliasing */}
      <SMAA />
    </EffectComposer>
  );
}
