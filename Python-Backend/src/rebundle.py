"""Rebuild the deliverable zip from current code + figures + result docs."""
import zipfile, os, glob
code = ['pidm_model.py', 'pidm_losses.py', 'pidm_diffusion.py', 'data_split.py',
        'prep_cache.py', 'prep_cache_of.py', 'train_pidm.py', 'train_diffusion.py',
        'finetune.py', 'evaluate_and_plot.py', 'compare_models.py', 'run_ablations.py',
        'fill_paper.py', 'fig_inference_visuals.py', 'fig_uncertainty.py', 'fig_diffusion.py',
        'xai_explain.py', 'architecture_diagram.py', 'uncertainty_calibration.py',
        'baseline_simvp.py', 'baseline_convlstm.py', 'make_sota_comparison.py', 'rebundle.py',
        'README_PIDM.md', 'requirements.txt', 'PAPER.md', 'PAPER_filled.md']
figs = sorted(glob.glob('results/*.png'))
docs = sorted(glob.glob('results/*.json')) + sorted(glob.glob('results/*.md'))
allf = [f for f in code + figs + docs if os.path.exists(f)]
out = 'ISRO_PIDM_bundle.zip'
if os.path.exists(out):
    os.remove(out)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in allf:
        z.write(f, arcname='ISRO_PIDM/' + f)
print(f'bundle {out}: {os.path.getsize(out)/1e6:.1f} MB, {len(allf)} files '
      f'({len([f for f in code if os.path.exists(f)])} code+docs, {len(figs)} figs)')
