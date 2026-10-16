#!/usr/bin/env python3
"""Publish the 44 identified BERT checkpoints from the author's Mac.
Standard library only. Original experiments are read-only. Re-run to resume.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = 'kazuoishii/PPS-ID'
TAG = 'bert-checkpoints-2026-10'
TARGET = '64b2ea1cfe1c7698f8fd13a821af94291b778a7c'
EXPECTED = {'sst2/seed_42/M0': 'b52e7de314e651ee23ab36f4c672ae882a6ce6238b9e9902b561bd51b16bba54', 'sst2/seed_42/recursive_M1': '567d764a568c6b026cebd6030692180b1123a104fc84d1c5cc753cc1d29d51e8', 'sst2/seed_42/recursive_M2': 'd4e0edb5d166ef542d720cf3ef2f3cbcd8e576bd7d9135acc67e54ab9b8ae187', 'sst2/seed_42/recursive_M3': '6840248ceb0196244a97e7b0fde5877f2a34ae5ac3bf1058230360ab2df44494', 'sst2/seed_42/recursive_M4': 'db07480a6c4c0fca9d36fb0ec65ff82a62a55138d1709bc332de852ba549b0c3', 'sst2/seed_42/recursive_M5': '1d4828c7acf41ad01879fedabe11bb08f2665dd0ea3ad6ba78653311f37445b4', 'sst2/seed_42/fixed_M1': '6903a58a4f191d66f62ae6cb93e7de6ba69245d5d2c85339d401b5bb24c1556d', 'sst2/seed_42/fixed_M2': '8f5000e8d3c5d101fd19bee64e775ba45938dc12ef469099cada73729ba1b6f9', 'sst2/seed_42/fixed_M3': 'd618bdfd1c31c3e3424c1a6a4c989142235a2107d147f29ac7362a58bcff0859', 'sst2/seed_42/fixed_M4': 'aec28247c629555ba89a989f0fe0915d47515461cd65481107238594ec7345b0', 'sst2/seed_42/fixed_M5': 'cdb0df9477cb42964db9a33b3a2c53df47e0715b14634377dff359a481000c2e', 'sst2/seed_43/M0': 'e2190372a7ed39413e3152b3f723e9214e43ac0cf1eefc7535a5351710a8b9ee', 'sst2/seed_43/recursive_M1': '042ae17d9fd4acbcb82c426c94fd2004dcf211f17656379ea9841da60322a157', 'sst2/seed_43/recursive_M2': 'e8c7d01ee0cbef86f1a48ddaf21c4a6a511dea48601c1b4983d530bce8bf65bb', 'sst2/seed_43/recursive_M3': 'f242bc72a739a7a2dcca10f31d6e36b9f0ccaa33d83b774742976a0692d4513c', 'sst2/seed_43/recursive_M4': '373304f4539badb040d3e8fdaf2dd6a24f981fe883cfbfa8949dee523705fa41', 'sst2/seed_43/recursive_M5': 'a1d32714a5a2b3770fb0fcaea7f9e8b4d26c396d25c7ed7320e0bd43adcccbe8', 'sst2/seed_43/fixed_M1': '0decba71c0fa724c87ab94f716afb873da958d06d5acf4273a1b9440bea0012c', 'sst2/seed_43/fixed_M2': '446e0e9895ad8792a174d2b37cb8d7c78aa4df02d2408584caff8209dd1e6440', 'sst2/seed_43/fixed_M3': 'e0075c039cad602f054de41b44fb64cc92b2f17dbde5e52d19852bcb8c567f3d', 'sst2/seed_43/fixed_M4': '14019c9f83cb78290908aab69c059f265fcba1be889ea015d0a2b0ed365ef0e0', 'sst2/seed_43/fixed_M5': '6a194a5bc77b1d38e5e63e55a4a0792eda6d4f208525c9cc3d43d1cd6c4384cb', 'imdb/seed_42/M0': '94c6f1df9bd29d4023ef53639e6caff851d2ab79ee3f36aa92af79df08c246c9', 'imdb/seed_42/recursive_M1': 'ab2b80426da533463b5602d4beeafe2cd4286ab58cc998cd17e29ceba88a9d36', 'imdb/seed_42/recursive_M2': '248e10c5b7e6730618782d325c88b502cfbad6bcd6e2228237d50da874cde649', 'imdb/seed_42/recursive_M3': '5364c5009d3d3ca3c08989acd6b841ebf21cd9a24b80439d576acc7c49d45d86', 'imdb/seed_42/recursive_M4': 'd934b42b57d1a3e46efea199c281ec43e9e152efc1f5f0a5264fcb8b80dd80af', 'imdb/seed_42/recursive_M5': '2cebef9ef1ab0f666401e1ea82cd3859224ff1592322b6e303445bd0a979bb14', 'imdb/seed_42/fixed_M1': '76d9ba1f56948c5d8900a0471d7a2ce8b854e889bf14a7cf6d5b2a00a2840ad8', 'imdb/seed_42/fixed_M2': '03ee5a8d9ff18dc2657878145ac441818b5746a73b809eb0f4eca50a34f9871a', 'imdb/seed_42/fixed_M3': '49498dc27a3c16c600db81394e2e9bf71867d95edb061f3d0278eed5f768368e', 'imdb/seed_42/fixed_M4': 'd25564e4502c51e4f4a79fd37a90ced1d694fbabdc58ad41d41cea37a8cd89e1', 'imdb/seed_42/fixed_M5': '7e70ab3f46232188abe129ef70e04d1ba98435eaffc0c5d4ea46ea17673d70b0', 'imdb/seed_43/M0': '5187c21f3986ba89ce844995388f29f20be69a9fb3a56100a65010b44675fedf', 'imdb/seed_43/recursive_M1': 'a99e10622966b5c2560effdc269a149be7853ed2c86d5e7d98a484a3b596b390', 'imdb/seed_43/recursive_M2': 'da30375330f9f5018652c214ec8ea47e86504acbbc0a83ada73b8675899d711c', 'imdb/seed_43/recursive_M3': '091be861480d8867becc0a9fdcb82c166a7b6434ad5ede5c214a8bc18b1f3a23', 'imdb/seed_43/recursive_M4': '576a6c338f31c2c20a9299855bb5bd9c3c5cc6a82cab83a019e3c844fd6087f0', 'imdb/seed_43/recursive_M5': '6dca1fde43013d5c6c5a2614281b448a479b89ecf4c7ba0968214b95fbcddf6f', 'imdb/seed_43/fixed_M1': '8159549b3b73af5b7be115ab0fdf431becc718e9c417b0ca281f0debd1f8dcdb', 'imdb/seed_43/fixed_M2': '98be2a26048998f1adb6d7efe54b90f41cbb7e86d0f1823fd8cf073c6d768473', 'imdb/seed_43/fixed_M3': '2a59dc7d46504f68c548388c3980953f13e1c2592b2376828b04ffed921fb026', 'imdb/seed_43/fixed_M4': '26f62903646d57e3ae167fb7f0eb07eb3c433ff17bb2694231a76d2bdd845d33', 'imdb/seed_43/fixed_M5': '81fca4375384a16ea9279b1b928398718402e67252481318ff3838e2754bc154'}  # Filled with SHA-256 values of the published prediction archives.


def gh(*args, check=True):
    p = subprocess.run(['gh', *args], text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr.strip())
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def release(missing_ok=False):
    # GitHub's /releases/tags endpoint does not resolve an unpublished draft.
    # Authenticated release listings include drafts; use their stable IDs.
    p = gh('api', '--paginate', '--slurp', f'repos/{REPO}/releases?per_page=100')
    matches = [r for page in json.loads(p.stdout) for r in page if r['tag_name'] == TAG]
    if len(matches) > 1:
        raise RuntimeError('Multiple releases have the requested tag; nothing was modified.')
    if matches:
        return matches[0]
    if missing_ok:
        return None
    raise RuntimeError('The checkpoint release was not found in the authenticated release listing.')


def assets(r):
    # gh --paginate yields arrays; --slurp wraps them into one JSON array.
    p = gh('api', '--paginate', '--slurp', f'repos/{REPO}/releases/{r["id"]}/assets?per_page=100')
    return {a['name']: a for page in json.loads(p.stdout) for a in page}


def verify_remote(asset, path, sha):
    if asset['state'] != 'uploaded' or asset['size'] != path.stat().st_size:
        raise RuntimeError(f'Incomplete/conflicting existing asset: {path.name}. Nothing was overwritten.')
    remote_digest = asset.get('digest')
    if remote_digest:
        if remote_digest != 'sha256:' + sha:
            raise RuntimeError(f'Hash conflict: {path.name}. Nothing was overwritten.')
    else:
        # Older servers may omit digests; verify the actual downloaded bytes.
        p = subprocess.Popen(['gh', 'api', '-H', 'Accept: application/octet-stream',
                              f'repos/{REPO}/releases/assets/{asset["id"]}'], stdout=subprocess.PIPE)
        h = hashlib.sha256()
        for chunk in iter(lambda: p.stdout.read(8 * 1024 * 1024), b''):
            h.update(chunk)
        if p.wait() or h.hexdigest() != sha:
            raise RuntimeError(f'Download verification failed: {path.name}')


def main():
    if not shutil.which('gh'):
        raise RuntimeError('Install gh and run gh auth login first.')
    login = gh('api', 'user', '--jq', '.login').stdout.strip()
    if login != 'kazuoishii':
        raise RuntimeError(f'Expected kazuoishii; current account is {login}.')
    home = Path.home()
    specs = []
    for dataset in ('sst2', 'imdb'):
        for seed in (42, 43):
            root = (home/'PPS_ID'/f'bert_5gen_seed{seed}_mps' if dataset == 'sst2'
                    else home/'Desktop'/'imdb_kd_5gen')
            for generation in ['M0'] + [f'{b}_M{g}' for b in ('recursive','fixed') for g in range(1,6)]:
                rel = Path('M0') if generation == 'M0' else Path(*generation.split('_'))
                folder = root/f'seed_{seed}'/(generation if dataset == 'sst2' else rel)
                model_dir = folder/'model' if dataset == 'sst2' else folder
                prediction = folder/('observations.npz' if dataset == 'sst2' else 'predictions.npz')
                key = f'{dataset}/seed_{seed}/{generation}'
                weight = model_dir/'model.safetensors'
                config = model_dir/'config.json'
                for f in (weight, config, prediction):
                    if not f.is_file() or f.is_symlink():
                        raise RuntimeError(f'Missing/non-regular file: {f}')
                if digest(prediction) != EXPECTED[key]:
                    raise RuntimeError(f'Prediction archive differs from the published experiment: {key}')
                cfg = json.loads(config.read_text())
                if cfg.get('model_type') != 'bert':
                    raise RuntimeError(f'Unexpected model configuration: {key}')
                specs.append((key, folder, model_dir, prediction, weight))
    if len(specs) != 44:
        raise RuntimeError('Expected 44 checkpoints.')
    print('Preflight passed: 44 weights/configs found; all paired prediction archives match the published experiments.', flush=True)
    print('This confirms saved-file pairing, not re-evaluation of model predictions.', flush=True)
    out = home/'Desktop'/'pps_weights_upload'
    out.mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < 2 * 1024**3:
        raise RuntimeError('At least 2 GiB of free disk space is needed for one temporary model archive.')
    notes = out/'CHECKPOINT_README.md'
    notes.write_text('''# PPS-ID BERT checkpoints\n\n44 checkpoints: SST-2 and IMDb, seeds 42/43, M0 and recursive/fixed M1--M5. Each tar contains a model.safetensors, the available model/tokenizer files, and its matching saved prediction archive. Pairing was checked by SHA-256 against the public revision prediction files. Model inference was not rerun by this publication script. CHECKPOINT_MANIFEST.json records file and archive hashes.\n\nExtract one tar into a separate directory; model files are in model/. Obtain datasets from their original providers. These are BERT checkpoints, not the 55 CIFAR-10 checkpoints needed for the reported CKA recomputation. The separately reported 100-update diagnostic raw logs are not included.\n\nAssociated analysis materials: https://github.com/kazuoishii/PPS-ID/tree/64b2ea1cfe1c7698f8fd13a821af94291b778a7c/revision_2026_10\n''')
    if release(missing_ok=True) is None:
        gh('release','create',TAG,'-R',REPO,'--draft','--target',TARGET,
           '--title','PPS-ID BERT checkpoints: SST-2 and IMDb','--notes-file',str(notes))
    r = release()
    known = assets(r)
    manifest = []
    allowed = {'model.safetensors','config.json','tokenizer.json','tokenizer_config.json',
               'special_tokens_map.json','vocab.txt','added_tokens.json','generation_config.json'}
    def upload(path):
        sha = digest(path)
        if path.name not in known:
            if not release()['draft']:
                raise RuntimeError('An already published release is incomplete; it was not modified.')
            gh('release','upload',TAG,str(path),'-R',REPO)
            known.update(assets(release()))
        verify_remote(known[path.name],path,sha)
        return sha
    for i,(key,folder,model_dir,prediction,weight) in enumerate(specs,1):
        asset_name = key.replace('/','_')+'.tar'
        archive = out/asset_name
        members = [(p,'model/'+p.name) for p in sorted(model_dir.iterdir()) if p.name in allowed and p.is_file()]
        if any(p.is_symlink() for p,_ in members):
            raise RuntimeError('Symlinks are not packaged.')
        members.append((prediction,prediction.name))
        member_info = []
        print(f'[{i}/44] Preparing/uploading {asset_name}',flush=True)
        with tarfile.open(archive,'w',format=tarfile.USTAR_FORMAT) as tar:
            for p,name in members:
                if p.stat().st_size >= 2 * 1024**3:
                    raise RuntimeError('An archive member exceeds the release asset size limit.')
                info = tarfile.TarInfo(name);info.size=p.stat().st_size;info.mode=0o644;info.mtime=0
                with p.open('rb') as f:tar.addfile(info,f)
                member_info.append({'path':name,'size_bytes':p.stat().st_size,'sha256':digest(p)})
        if archive.stat().st_size >= 2 * 1024**3:
            raise RuntimeError('Archive exceeds the release asset size limit.')
        sha = upload(archive)
        manifest.append({'model':key,'asset':asset_name,'size_bytes':archive.stat().st_size,'sha256':sha,'files':member_info})
        archive.unlink()  # Delete only the temporary tar created above, after remote verification.
        print(f'[{i}/44] Verified {asset_name}',flush=True)
    manifest_path = out/'CHECKPOINT_MANIFEST.json'
    manifest_path.write_text(json.dumps({'repository':REPO,'tag':TAG,'checkpoints':manifest},indent=2)+'\n')
    upload(manifest_path);upload(notes)
    expected = {x['asset'] for x in manifest} | {manifest_path.name,notes.name}
    final = assets(release())
    if not expected.issubset(final):
        raise RuntimeError('Final asset inventory is incomplete.')
    if release()['draft']:
        gh('release','edit',TAG,'-R',REPO,'--draft=false','--latest=false')
    r = release()
    if r['draft']:
        raise RuntimeError('Release is still a draft.')
    receipt = {'url':r['html_url'],'tag':TAG,'checkpoint_assets':44,'metadata_assets':2,
               'verification':'all asset sizes and SHA-256 hashes verified',
               'scope':'BERT checkpoints only; no CIFAR-10 checkpoints or original 100-update logs'}
    (out/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PUBLICATION COMPLETE',flush=True)
    print(json.dumps(receipt,indent=2),flush=True)

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError,OSError,ValueError,subprocess.SubprocessError) as e:
        print('STOPPED:',str(e),file=sys.stderr)
        print('Original files were not modified. Re-run the same command to resume after resolving the error.',file=sys.stderr)
        sys.exit(1)
