"""Order 6: reuse saved perfect, middle-vacancy, end-vacancy and four-sphere runs.

    python scripts/compare_vacancies_v22.py --perfect chain5_80_ex --middle chain5_missing_middle_ex --end chain5_missing_end_ex --four chain4_regular_ex --name vacancies_ex

Local gap probes x=-40,+40 nm compare perfect/middle/end at fixed positions.
The regular four-sphere chain occupies those points, so all four cases are
instead compared 40 nm from their own left/right end-sphere centres
(20 nm beyond the surfaces). The end
vacancy and regular four-sphere chain are also compared after translation.
These translated differences diagnose numerical consistency, not a separate
physical defect effect. All original simulation folders are read only.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import inspect
import json
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1]/'src'
if not (_LOCAL_SOURCE/'surface_plasmons/__init__.py').is_file():
    raise SystemExit('Place this file in scripts/ inside the existing project.')
sys.path.insert(0,str(_LOCAL_SOURCE))

try:
    from . import compare_particle_counts_v19 as shared
    from .compare_central_displacements_v21 import segment_is_exterior, sample_at, _csv
except ImportError:
    import compare_particle_counts_v19 as shared
    from compare_central_displacements_v21 import segment_is_exterior, sample_at, _csv

if 'expected_vacancy_index' not in inspect.signature(shared.read_run).parameters:
    raise SystemExit('Install the updated compare_particle_counts_v19.py from the order-6 download.')

import numpy as np
from surface_plasmons.project import make_simulation_config
from surface_plasmons.results import safe_simulation_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = ('perfect','middle','end','four')
LABELS = {'perfect':'Perfect five','middle':'Middle missing','end':'Right end missing','four':'Regular four'}
COMPONENTS = ('Ex','Ey','Ez')
EXPECTED_X_NM = {'perfect':[-160,-80,0,80,160], 'middle':[-160,-80,80,160],
                 'end':[-160,-80,0,80], 'four':[-120,-40,40,120]}


def _record(run,wavelength,x,field,baseline,frame,probe):
    row = dict(case=run['case'], nominal_particle_count=run['count'], active_particle_count=run['active_count'],
               missing_particle_index=run['vacancy_index'], wavelength_nm=float(wavelength),
               coordinate_frame=frame, probe=probe, x_nm=float(x*1000), y_nm=20., z_nm=0.,
               distance_from_end_centre_nm=40. if frame=='end_relative' else None,
               distance_from_end_surface_nm=20. if frame=='end_relative' else None,
               source_result_folder=str(run['directory']))
    for j,c in enumerate(COMPONENTS):
        row.update({f'S_{c}_real':float(field[j].real),f'S_{c}_imag':float(field[j].imag),
                    f'S_{c}_magnitude':float(abs(field[j])),
                    f'delta_S_{c}_real':float((field[j]-baseline[j]).real),
                    f'delta_S_{c}_imag':float((field[j]-baseline[j]).imag)})
    scale=float(np.linalg.norm(baseline))
    row['relative_complex_field_change_from_perfect']=float(np.linalg.norm(field-baseline)/scale) if scale>1e-20 else None
    return row


def _end_view(run):
    """An x translation changes no incident phase for propagation along -y."""
    coordinates=run['coordinates'].copy();coordinates[:,0]-=run['first_x']
    centres=run['centres'].copy();centres[:,0]-=run['first_x']
    return {**run,'coordinates':coordinates,'centres':centres,'first_x':0.}


def compare_vacancies(runs):
    if [r['case'] for r in runs]!=list(CASES):
        raise ValueError('Supply cases in order: perfect, middle, end, four.')
    grid=runs[0]['grid'];pol=runs[0]['config']['source']['polarisation']
    reference=make_simulation_config(runs[0]['config'],experiment_directory='/tmp/vacancy_compare_preview',stage_name='frequency_scan/particle')
    for run in runs:
        expected=np.array(EXPECTED_X_NM[run['case']])/1000
        if (len(run['centres'])!=len(expected) or not np.allclose(run['centres'][:,0],expected,rtol=0,atol=1e-12)
                or not np.allclose(run['centres'][:,1:],[.02,0],rtol=0,atol=1e-12)):
            raise ValueError(f"{run['directory']}: occupied positions do not match {run['case']}.")
        if run['config']['source']['polarisation']!=pol:
            raise ValueError('Compare one polarisation at a time.')
        if run['grid'].shape!=grid.shape or not np.allclose(run['grid'],grid,rtol=1e-10,atol=1e-6):
            raise ValueError('Saved wavelength grids differ.')
        raw=make_simulation_config(run['config'],experiment_directory='/tmp/vacancy_compare_preview',stage_name='frequency_scan/particle')
        if (not np.allclose(raw['cell_size_um'],reference['cell_size_um'],rtol=0,atol=1e-12)
                or not np.isclose(raw['pml_thickness_um'],reference['pml_thickness_um'])):
            raise ValueError('The comparison requires the common five-sphere cell and PML settings.')
        for key in ('field_region_center_um','field_region_size_um'):
            if not np.allclose(raw['output'][key],reference['output'][key],rtol=0,atol=1e-12):
                raise ValueError('The comparison requires the common five-sphere line monitor.')
    resolution=min(r['resolution'] for r in runs)
    local_runs=runs[:3]
    lower=max(float(r['coordinates'][:,0].min()) for r in local_runs)+2/resolution
    upper=min(float(r['coordinates'][:,0].max()) for r in local_runs)-2/resolution
    xs=[float(x) for x in np.unique(runs[0]['coordinates'][:,0]) if lower<=x<=upper
        and segment_is_exterior(float(x),float(x),local_runs,resolution,extra_cells=1)]
    if len(xs)<4:
        raise ValueError('Too few common exterior positions.')
    local_rows=[];profile_rows=[];end_rows=[];translation_rows=[]
    end_view,four_view=_end_view(runs[2]),_end_view(runs[3])
    views=[end_view,four_view]
    lo=max(float(r['coordinates'][:,0].min()) for r in views)+2/resolution
    hi=min(float(r['coordinates'][:,0].max()) for r in views)-2/resolution
    us=[float(x) for x in np.unique(four_view['coordinates'][:,0]) if lo<=x<=hi
        and segment_is_exterior(float(x),float(x),views,resolution,extra_cells=1)]
    if len(us)<4:
        raise ValueError('Too few common translated positions.')
    for wave in grid:
        for points,records,probe in (([-.04,.04],local_rows,'fixed_gap'),(xs,profile_rows,'profile')):
            for x in points:
                fields=[sample_at(r,wave,x,local_runs,resolution) for r in local_runs]
                for r,value in zip(local_runs,fields):
                    records.append(_record(r,wave,x,value,fields[0],'laboratory',probe))
        for side in ('left','right'):
            baseline_x=runs[0]['first_x']-.04 if side=='left' else float(runs[0]['centres'][:,0].max())+.04
            baseline=sample_at(runs[0],wave,baseline_x,[runs[0]],resolution)
            for run in runs:
                x=run['first_x']-.04 if side=='left' else float(run['centres'][:,0].max())+.04
                field=sample_at(run,wave,x,[run],resolution)
                end_rows.append(_record(run,wave,x,field,baseline,'end_relative',side))
        for u in us:
            end_field=sample_at(end_view,wave,u,views,resolution)
            four_field=sample_at(four_view,wave,u,views,resolution)
            row=dict(wavelength_nm=float(wave),x_from_leftmost_centre_nm=u*1000,
                     end_original_x_nm=(u+runs[2]['first_x'])*1000,
                     regular4_original_x_nm=(u+runs[3]['first_x'])*1000)
            for j,c in enumerate(COMPONENTS):
                row.update({f'end_{c}_real':float(end_field[j].real),f'end_{c}_imag':float(end_field[j].imag),
                            f'regular4_{c}_real':float(four_field[j].real),f'regular4_{c}_imag':float(four_field[j].imag)})
            scale=float(np.linalg.norm(four_field))
            row['relative_complex_translation_difference']=float(np.linalg.norm(end_field-four_field)/scale) if scale>1e-20 else None
            translation_rows.append(row)
    return dict(local=local_rows,profiles=profile_rows,ends=end_rows,translation=translation_rows)


def save_comparison(runs,data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():
        raise ValueError(f'Comparison folder already exists: {output}. Choose a new --name.')
    output.mkdir(parents=True,exist_ok=False)
    for key,name in [('local','local_gap_fields.csv'),('profiles','common_exterior_fields.csv'),
                     ('ends','end_relative_fields.csv'),('translation','end_vs_four_translation.csv')]:
        _csv(output/name,data[key])
    pol=runs[0]['config']['source']['polarisation'];grid=runs[0]['grid']
    mid=float(grid[np.argmin(abs(grid-1100))])
    waves=sorted({float(grid[0]),mid,float(grid[-1])})
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,x in enumerate((-40.,40.)):
        for case in CASES[:3]:
            rows=sorted([r for r in data['local'] if r['case']==case and np.isclose(r['x_nm'],x)],key=lambda r:r['wavelength_nm'])
            for j,part in enumerate(('real','imag')):
                axes[i,j].plot([r['wavelength_nm'] for r in rows],[r[f'S_{pol}_{part}'] for r in rows],label=LABELS[case])
                axes[i,j].set(xlabel='Vacuum wavelength (nm)',ylabel=f'{part.capitalize()}(scattered {pol} / Eref)',title=f'Fixed exterior gap probe x={x:g} nm')
                axes[i,j].grid(alpha=.2);axes[i,j].legend()
    fig.tight_layout();fig.savefig(output/'local_complex_spectra.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for side,axis in zip(('left','right'),axes):
        for wave in waves:
            rows=[next(r for r in data['ends'] if r['case']==case and r['probe']==side and r['wavelength_nm']==wave) for case in CASES]
            axis.plot(range(4),[r[f'S_{pol}_magnitude'] for r in rows],'o-',label=f'{wave:.1f} nm')
        axis.set(xticks=range(4),xticklabels=['Perfect 5','Middle missing','End missing','Regular 4'],ylabel=f'|scattered {pol} / Eref|',title=f'40 nm from {side} end centre (20 nm outside surface)')
        axis.tick_params(axis='x',labelrotation=15);axis.grid(alpha=.2);axis.legend()
    fig.tight_layout();fig.savefig(output/'end_response_by_case.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for case in CASES[:3]:
        rows=sorted([r for r in data['profiles'] if r['case']==case and r['wavelength_nm']==mid],key=lambda r:r['x_nm'])
        for axis,part in zip(axes,('real','imag')):
            xvalues=[];yvalues=[];previous=None
            for row in rows:
                if previous is not None and row['x_nm']-previous>1500/min(r['resolution'] for r in runs):
                    xvalues.append(np.nan);yvalues.append(np.nan)
                xvalues.append(row['x_nm']);yvalues.append(row[f'S_{pol}_{part}']);previous=row['x_nm']
            axis.plot(xvalues,yvalues,'.-',markersize=3,label=LABELS[case])
            axis.set(xlabel='Fixed laboratory x (nm)',ylabel=f'{part.capitalize()}(scattered {pol} / Eref)')
            axis.grid(alpha=.2);axis.legend()
    fig.suptitle(f'Common exterior profiles at {mid:.1f} nm')
    fig.tight_layout();fig.savefig(output/'common_exterior_profiles.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rows=sorted([r for r in data['translation'] if r['wavelength_nm']==mid],key=lambda r:r['x_from_leftmost_centre_nm'])
    for axis,part in zip(axes,('real','imag')):
        for key,label,style in [('end','End vacancy','-'),('regular4','Regular four','--')]:
            xvalues=[];yvalues=[];previous=None
            for row in rows:
                x=row['x_from_leftmost_centre_nm']
                if previous is not None and x-previous>1500/min(r['resolution'] for r in runs):
                    xvalues.append(np.nan);yvalues.append(np.nan)
                xvalues.append(x);yvalues.append(row[f'{key}_{pol}_{part}']);previous=x
            axis.plot(xvalues,yvalues,style,label=label)
        axis.set(xlabel='x from leftmost sphere centre (nm)',ylabel=f'{part.capitalize()}(scattered {pol} / Eref)')
        axis.grid(alpha=.2);axis.legend()
    fig.suptitle(f'Translation consistency at {mid:.1f} nm; equivalent four-sphere arrangements')
    fig.tight_layout();fig.savefig(output/'end_vs_four_after_translation.png',dpi=180);plt.close(fig)
    end_vectors=np.array([[complex(r[f'end_{c}_real'],r[f'end_{c}_imag']) for c in COMPONENTS] for r in data['translation']])
    four_vectors=np.array([[complex(r[f'regular4_{c}_real'],r[f'regular4_{c}_imag']) for c in COMPONENTS] for r in data['translation']])
    scale=float(np.linalg.norm(four_vectors))
    cells={r['case']:make_simulation_config(r['config'],experiment_directory=output,stage_name='summary')['cell_size_um'] for r in runs}
    summary=dict(nominal_spacing_nm=80,particle_radius_nm=20,polarisation=pol,
                 fixed_gap_probes_nm=[-40,40],fixed_gap_cases=list(CASES[:3]),
                 fixed_gap_exclusion='The regular four-sphere control has spheres at x=-40,+40 nm and is omitted from these exterior probes.',
                 end_probes='40 nm from each actual left/right end-sphere centre, 20 nm outside its surface; coordinates differ between arrangements',
                 translation_applied_to_regular4_nm=(runs[2]['first_x']-runs[3]['first_x'])*1000,
                 translation_relative_l2_difference=float(np.linalg.norm(end_vectors-four_vectors)/scale) if scale>1e-20 else None,
                 translation_interpretation='The end vacancy and regular four-sphere array are physically equivalent after x translation. Differences after alignment are numerical consistency diagnostics; the finite PML cells and mirror setups do not translate with the plotted coordinates. Review domain/resolution/time convergence.',
                 numerical_acceptance_note='These graphs do not establish numerical convergence or electrostatic applicability.',
                 no_new_simulations_started=True,
                 cases=[dict(case=r['case'],nominal_particle_count=r['count'],active_particle_count=r['active_count'],
                             missing_particle_index=r['vacancy_index'],result_folder=str(r['directory']),
                             sphere_x_nm=(r['centres'][:,0]*1000).tolist(),cell_size_um=cells[r['case']],
                             mirror_axes=r['config']['numerics'].get('mirror_axes',['x','z'] if r['config']['numerics'].get('mirror_symmetry') else []),
                             **r['statuses']) for r in runs])
    (output/'comparison_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    for case in CASES:
        parser.add_argument('--'+case,required=True,help='Completed simulation name or directory.')
    parser.add_argument('--name',required=True)
    parser.add_argument('--results-root',type=Path,default=PROJECT_ROOT/'data/results')
    parser.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'data/comparisons')
    args=parser.parse_args(argv)
    runs=[]
    for case in CASES:
        value=getattr(args,case);path=Path(value).expanduser()
        directory=path.resolve() if path.is_dir() else args.results_root.expanduser().resolve()/safe_simulation_name(value)
        run=shared.read_run(directory,4 if case=='four' else 5,
                            expected_vacancy_index=2 if case=='middle' else 4 if case=='end' else None)
        run['case']=case;runs.append(run)
    data=compare_vacancies(runs)
    output=args.output_root.expanduser().resolve()/safe_simulation_name(args.name)
    save_comparison(runs,data,output)
    print(f'Saved vacancy comparison: {output}')
    print('The end-vacancy/four-sphere alignment is a numerical consistency comparison.')
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        raise SystemExit(f'Vacancy comparison failed: {exc}') from exc
