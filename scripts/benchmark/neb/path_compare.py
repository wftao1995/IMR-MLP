"""Conservative comparisons of labelled, unfolded NEB paths (no EFS)."""
from dataclasses import dataclass
import numpy as np
from ase.geometry import find_mic


@dataclass(frozen=True)
class Tolerances:
    rms: float = .01
    maximum: float = .03
    cell: float = 1e-4
    energy: float = 1e-4


def mask(atoms):
    result = np.ones((len(atoms), 3), dtype=bool)
    for constraint in atoms.constraints:
        name = type(constraint).__name__
        if name == 'FixAtoms': result[constraint.get_indices()] = False
        elif name == 'FixScaled': result[constraint.get_indices()] &= ~np.asarray(constraint.mask)
        else: raise ValueError('Unsupported path constraint: '+name)
    return result


def _aligned(a, b):
    if len(a)<3 or len(b)<3: raise ValueError('Complete paths require at least three images')
    numbers=a[0].numbers
    for frame in [*a,*b]:
        if not np.array_equal(frame.numbers,numbers) or not np.array_equal(frame.pbc,a[0].pbc):
            raise ValueError('Different labelled species or periodic boundaries')
        if not np.array_equal(mask(frame),mask(a[0])): raise ValueError('Different fixed-coordinate mappings')
    ac=np.array([x.cell.array for x in a]);bc=np.array([x.cell.array for x in b])
    # A single positive unimodular row-basis change, fixed for the entire path.
    u=bc[0]@np.linalg.inv(ac[0]);integer=np.rint(u)
    if not np.allclose(u,integer,atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(integer),1,atol=1e-8):
        raise ValueError('Endpoint lattices are not the same oriented periodic lattice')
    if not all(a[0].pbc) and not np.array_equal(integer,np.eye(3)):
        raise ValueError('Partial-PBC basis mixing is not a supported equivalence')
    bc=np.einsum('ab,ibc->iac',np.linalg.inv(integer),bc)
    ap=np.array([x.positions for x in a]);bp=np.array([x.positions for x in b])
    # Infer the anchor atom's constant lattice gauge from both endpoints when
    # the cell changes; subtracting its initial position alone would introduce
    # a spurious affine displacement. Null cell-change modes are absorbed in
    # the single whole-path origin. Ambiguous noninteger solutions fail closed.
    start=bp[0,0]-ap[0,0];end=bp[-1,0]-ap[-1,0]
    anchor=np.rint(np.linalg.lstsq((ac[-1]-ac[0]).T,end-start,rcond=None)[0])
    anchor[~a[0].pbc]=0
    origin=start-anchor@ac[0]
    bp=bp-origin
    common_origin=ap[0,0].copy();ap=ap-common_origin;bp=bp-common_origin
    delta=bp[0]-ap[0];nearest,_=find_mic(delta,ac[0],a[0].pbc)
    shifts=np.rint((delta-nearest)@np.linalg.inv(ac[0]))
    shifts[:,~a[0].pbc]=0
    bp-=np.einsum('na,iab->inb',shifts,bc)
    if max(np.abs(ap[0]-bp[0]).max(),np.abs(ap[-1]-bp[-1]).max(),np.abs(ac[-1]-bc[-1]).max())>1e-7:
        raise ValueError('Different labelled endpoints or periodic winding')
    return ap,ac,bp,bc


def _coordinates(pos,cells,reference,settings):
    if not settings.get('vcneb',False): return pos.reshape(len(pos),-1)
    inv=np.linalg.inv(reference);F=np.einsum('ab,ibc->iac',inv,cells)
    C=F@F.transpose(0,2,1)
    cell=C[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
    scales=np.asarray(settings['cell_scales'])
    frac=np.einsum('ina,iab->inb',pos,np.linalg.inv(cells))
    atoms=frac@reference if settings.get('recast_atomic_force',True) else frac
    return np.concatenate([cell*scales,atoms.reshape(len(pos),-1)],axis=1)


def _nodes(q):
    steps=np.linalg.norm(np.diff(q,axis=0),axis=1)
    s=np.r_[0,np.cumsum(steps)]
    if s[-1]<=1e-14: return np.linspace(0,1,len(q))
    return s/s[-1]


def _sample(s,values,nodes):
    # Coincident segments carry no geometric length. Keep one comparison knot.
    unique,index=np.unique(s,return_index=True)
    values=values.reshape(len(values),-1)[index]
    return np.array([np.interp(nodes,unique,col) for col in values.T]).T


def compare_paths(a,b,*,reference=None,settings=None,tolerances=Tolerances()):
    """Compare without atom matching, image reversal or changing either input."""
    settings=settings or {}
    try:
        ap,ac,bp,bc=_aligned(a,b)
        reference=ac[0] if reference is None else np.asarray(reference)
        sa=_nodes(_coordinates(ap,ac,reference,settings));sb=_nodes(_coordinates(bp,bc,reference,settings))
        nodes=np.unique(np.r_[sa,sb])
        pa=_sample(sa,ap,nodes).reshape(len(nodes),-1,3);pb=_sample(sb,bp,nodes).reshape(len(nodes),-1,3)
        ca=_sample(sa,ac,nodes).reshape(-1,3,3);cb=_sample(sb,bc,nodes).reshape(-1,3,3)
        distances=np.linalg.norm(pa-pb,axis=2)
        rms=float(np.sqrt(np.trapezoid(np.mean(distances**2,axis=1),nodes)))
        maximum=float(distances.max());cell=float(np.linalg.norm(np.linalg.solve(ca,cb)-np.eye(3),axis=(1,2)).max())
        return dict(equivalent=rms<=tolerances.rms and maximum<=tolerances.maximum and cell<=tolerances.cell,
                    rms_A=rms,max_A=maximum,relative_cell=cell,comparison_knots=len(nodes))
    except (ValueError,np.linalg.LinAlgError) as error:
        return dict(equivalent=False,reason=str(error))


def geometry_relation(a,b,*,reference=None,settings=None,tolerances=Tolerances(),symprec=1e-5):
    """Annotate one fixed space operation/permutation; never merge or reorder stored paths."""
    import spglib
    if not all(b[0].pbc):return None
    data=spglib.get_symmetry((b[0].cell.array,b[0].get_scaled_positions(wrap=False),b[0].numbers),symprec=symprec)
    if data is None:return None
    source=b[0].get_scaled_positions(wrap=False);n=len(source);fixed=mask(b[0])
    for rotation,translation in zip(data['rotations'],data['translations']):
        permutation=[]
        transformed=source@rotation.T+translation
        for i in range(n):
            delta=(transformed[i]-source)@b[0].cell.array
            _,dist=find_mic(delta,b[0].cell,b[0].pbc)
            candidates=np.flatnonzero((dist<symprec)&(b[0].numbers==b[0].numbers[i]))
            if len(candidates)!=1:break
            permutation.append(int(candidates[0]))
        if len(permutation)!=n or len(set(permutation))!=n:continue
        if np.array_equal(rotation,np.eye(3)) and permutation==list(range(n)):continue
        # Constraints and all cell metrics must admit this same operation.
        if not np.array_equal(fixed,fixed[permutation]):continue
        rotated=[];valid=True
        for frame in b:
            cell=frame.cell.array;metric=cell@cell.T
            if np.linalg.norm(rotation.T@metric@rotation-metric)>symprec*max(1,np.linalg.norm(cell)):
                valid=False;break
            # The same operation must preserve allowed coordinate directions,
            # including anisotropic selective masks, in the source chart.
            chart=cell
            if (settings or {}).get('vcneb',False):
                chart=np.asarray(reference) if (settings or {}).get('recast_atomic_force',True) else np.eye(3)
            action=np.linalg.solve(chart,rotation.T@chart)
            if any(not np.allclose(action@np.diag(row),np.diag(row)@action,atol=1e-8,rtol=0) for row in fixed):
                valid=False;break
            sample=frame.copy();positions=(frame.get_scaled_positions(wrap=False)@rotation.T+translation)@cell
            sample.positions[np.array(permutation)]=positions;rotated.append(sample)
        if not valid:continue
        comparison=compare_paths(a,rotated,reference=reference,settings=settings,tolerances=tolerances)
        if comparison['equivalent']:
            return dict(rotation=rotation.tolist(),translation=translation.tolist(),permutation=permutation,comparison=comparison)
    return None


def classify(records,paths,*,reference=None,settings=None,tolerances=Tolerances()):
    ranked=sorted((k for k,v in records.items() if v['status']=='converged'),key=lambda k:(records[k]['barrier_eV'],k))
    groups=[];relations=[]
    for key in ranked:
        for group in groups:
            if all(abs(records[key]['barrier_eV']-records[other]['barrier_eV'])<=tolerances.energy and
                   compare_paths(paths[key],paths[other],reference=reference,settings=settings,tolerances=tolerances)['equivalent'] for other in group):
                group.append(key);break
        else:groups.append([key])
    for i,left in enumerate(groups):
        for right in groups[i+1:]:
            a,b=left[0],right[0]
            if abs(records[a]['barrier_eV']-records[b]['barrier_eV'])>tolerances.energy:continue
            relation=geometry_relation(paths[a],paths[b],reference=reference,settings=settings,tolerances=tolerances)
            if relation:relations.append(dict(left=a,right=b,**relation))
    baseline=records.get('baseline',{});comparisons={}
    for key in ranked:
        if key=='baseline' or baseline.get('status')!='converged':continue
        delta=records[key]['barrier_eV']-baseline['barrier_eV']
        comparisons[key]=dict(delta_eV=delta,lower=delta < -tolerances.energy,
            needs_refinement=(abs(delta)<=5*tolerances.energy or
                max(records[key]['residual'],baseline['residual'])>=.8*records[key]['force_tolerance']))
    return dict(ranking=ranked,duplicate_groups=groups,geometric_relations=relations,baseline_comparisons=comparisons,
                incomplete=[k for k,v in records.items() if v['status']!='converged'])
