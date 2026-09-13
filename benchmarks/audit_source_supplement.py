#!/usr/bin/env python3
"""Build and inspect a lossless source supplement; no DB/model/network calls."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_root',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    b=args.run_root.resolve();o=args.output.resolve();o.mkdir(exist_ok=False)
    paths={'package':b/'full500-corrected-measurement-package-001/package.json',
           'cache':b/'integrity-audit-001/corrected-cache-v2/longmemeval_s.json',
           'annotations':b/'raw-vocabulary-audit-002/results.json',
           'candidate':Path('AgentMem-OS/benchmarks/source_packet_supplement.py').resolve(),
           'ranker':Path('AgentMem-OS/benchmarks/uncapped_lexical_retrieval.py').resolve(),
           'runner':Path(__file__).resolve()}
    inputs={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}
    policy={'status':'FROZEN_BEFORE_OUTPUTS','inputs':inputs,'character_cap':40000,'extra_character_cap':4000,
            'mechanism':'Preserve complete baseline packet; append uncapped-TFIDF-ranked complete source turns with role attribution into spare capacity only.',
            'not_single_variable_rank_ablation':'Adds context tokens within existing40000charactercap; does not replace existing ranking. New rank vocabulary and deduplication feed only the supplement.',
            'gate':{'exact_prefix_count':500,'stable_miss_gain_questions_min':10,'all500_annotated_turn_losses_max':0,
                    'every_added_turn_is_complete_scoped_source':True,'every_packet_within_caps':True},
            'paid_execution':'NOT_AUTHORIZED; evidence coverage does not establish QA gain or no answer regressions.'}
    (o/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    def deny(*args,**kwargs):raise RuntimeError('Offline audit: network prohibited')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    from agentmem_os.benchmarks.source_packet_supplement import supplement_packet
    package=json.loads(paths['package'].read_text());cache=json.loads(paths['cache'].read_text())
    memories={m['mid']:m for m in cache['memories']};queries={q['question_id']:q for q in cache['queries']}
    observed={r['question_id']:r for r in json.loads(paths['annotations'].read_text())}
    (o/'contexts').mkdir();(o/'receipts').mkdir();rows=[];start=time.monotonic()
    for i,c in enumerate(package['cases'],1):
        qid=c['id'];q=queries[qid]
        turns=[t for key in q['scope_keys'] for t in memories[key]['turns']]
        packet,receipts=supplement_packet(c['context'],turns,c['question'])
        # All assertions here are independent of rank selection or annotation-based scores.
        assert packet.startswith(c['context']) and len(packet)<=40000 and len(packet)-len(c['context'])<=4000
        for r in receipts:
            body=packet[r['packet_start']:r['packet_end']]
            assert hashlib.sha256(body.encode()).hexdigest()==r['source_sha256']
            assert any(t['content']==body and t['role']==r['role'] for t in turns)
        annotations=observed[qid]['annotated_turns']
        gained=[a for a in annotations if not a['baseline_exact_turn'] and a['text'] in packet]
        lost=[a for a in annotations if a['baseline_exact_turn'] and a['text'] not in packet]
        row={'question_id':qid,'stable_miss':observed[qid]['stable_miss'],'stable_pass':observed[qid]['stable_pass'],
             'abstention':c['abst'],'baseline_sha256':c['context_sha256'],'candidate_sha256':hashlib.sha256(packet.encode()).hexdigest(),
             'prefix_preserved':packet.startswith(c['context']),'extra_chars':len(packet)-len(c['context']),
             'added_turns':len(receipts),'gained_annotated_turns':len(gained),'lost_annotated_turns':len(lost),
             'gained_source_turns':[{'source_key':a['source_key'],'turn_index':a['turn_index']} for a in gained]}
        (o/'contexts'/f'{qid}.txt').write_text(packet)
        (o/'receipts'/f'{qid}.json').write_text(json.dumps(receipts,indent=2)+'\n');rows.append(row)
        if i%100==0:print(json.dumps({'completed':i,'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
    def summary(rs):return {'questions':len(rs),'changed':sum(r['extra_chars']>0 for r in rs),
        'questions_gaining_annotated_turns':sum(r['gained_annotated_turns']>0 for r in rs),
        'gained_annotated_turns':sum(r['gained_annotated_turns'] for r in rs),
        'lost_annotated_turns':sum(r['lost_annotated_turns'] for r in rs),
        'extra_chars_mean':sum(r['extra_chars'] for r in rs)/len(rs),'added_turns':sum(r['added_turns'] for r in rs)}
    result={k:summary(rs) for k,rs in {'all500':rows,'stable_miss':[r for r in rows if r['stable_miss']], 'stable_pass':[r for r in rows if r['stable_pass']]}.items()}
    result['checks']={'all500_prefix_preserved':len(rows)==500 and all(r['prefix_preserved'] for r in rows),
        'stable_miss_gain_min10':result['stable_miss']['questions_gaining_annotated_turns']>=10,
        'no_annotated_turn_loss':result['all500']['lost_annotated_turns']==0,
        'all_source_receipts_and_character_caps':True,
        'input_preservation':all(sha(p)==inputs[k]['sha256'] for k,p in paths.items())}
    result['offline_gate_pass']=all(result['checks'].values());result['paid_calls']=0
    (o/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(o/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
