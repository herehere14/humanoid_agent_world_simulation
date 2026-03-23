/** Post-processing pipeline.
 *
 * The current @react-three/postprocessing stack crashes at runtime in this
 * environment while mounting the composer, leaving the whole viewer blank.
 * Keep this component as a no-op so the world stays usable until the package
 * versions are aligned and the effect chain can be re-enabled safely.
 */

export function PostProcessing() {
  return null;
}
