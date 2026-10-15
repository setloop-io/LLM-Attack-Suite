"""Reference-backed security arguments scoped to declared adversaries.

This is a threat-model record, not a TLS scanner, cryptographic proof, or score.
It cannot close deployment coverage. Conditions are operator assertions, not
observations made by this module. No network traffic or secret values are read.
"""
import argparse
import json
from pathlib import Path
from .evidence import EvidenceLogger, file_sha256

REFERENCES={
 'tls13':{'title':'RFC8446: TLS1.3, sections1,8 and AppendixE','url':'https://www.rfc-editor.org/rfc/rfc8446.html'},
 'kdf':{'title':'NIST SP800-108 Rev1 Update1: PRF-based key derivation','url':'https://csrc.nist.gov/pubs/sp/800/108/r1/upd1/final'},
 'hkdf':{'title':'RFC5869: HMAC-based extract-and-expand KDF','url':'https://www.rfc-editor.org/rfc/rfc5869.html'}}


def assess(mechanism,adversary,conditions):
    if mechanism not in ('tls13','kdf','ratchet'): raise ValueError('unknown mechanism')
    if adversary not in ('passive_network','endpoint','pretermination','insider','passive_seam'): raise ValueError('unknown adversary')
    allowed={'tls13':{'tls13','authenticated_peers','uncompromised_keys','trusted_implementation','zero_rtt'},
             'kdf':{'standard_hmac_kdf','high_entropy_secret','domain_separation','secret_uncompromised'},
             'ratchet':{'separated_epoch_keys','enforced_epoch_budget','master_retained','same_key_pooling_only'}}[mechanism]
    if not isinstance(conditions,dict) or set(conditions)-allowed or any(type(v) is not bool for v in conditions.values()):
        raise ValueError('unknown condition or nonboolean assertion')
    result={'schema':'llm_attacker.security_assumption.v1','mechanism':mechanism,'adversary':adversary,
            'declared_conditions':conditions,'condition_evidence':'operator_assertions_not_independently_verified',
            'deployment_verified':False,'counts_as_attack_coverage':False,'counts_as_privacy_certificate':False,
            'checked_date':'2026-09-08'}
    if mechanism=='tls13':
        required={'tls13','authenticated_peers','uncompromised_keys','trusted_implementation'}
        result['references']=[REFERENCES['tls13']]
        result['claim']='Application-content confidentiality against a passive network observer under the stated TLS assumptions.'
        result['exclusions']=['Endpoint/pre-termination plaintext access','Packet lengths, timing and traffic analysis',
                              'Implementation compromise or key theft','0-RTT replay resistance and full forward-secrecy guarantees']
        if adversary!='passive_network': result['status']='OUTSIDE_PROTECTED_ADVERSARY_SCOPE'
        elif not all(conditions.get(k) is True for k in required): result['status']='ASSUMPTIONS_UNESTABLISHED'
        else:result['status']='REFERENCE_BACKED_CONDITIONAL_ARGUMENT'
    elif mechanism=='kdf':
        result['references']=[REFERENCES['kdf'],REFERENCES['hkdf']]
        result['claim']='Standard HMAC-based derivation relies on its specified PRF/extraction assumptions and sound secret handling.'
        result['exclusions']=['A raw SHA256(domain||secret||counter) construction is not certified by HMAC-KDF guidance.',
                              'Preimage resistance alone is not a general PRF/KDF proof.',
                              'Output width does not increase input-secret entropy.',
                              'Seed disclosure, context/epoch reuse, and retained-master compromise remain system risks.']
        if conditions.get('standard_hmac_kdf') is not True:result['status']='CUSTOM_CONSTRUCTION_REVIEW_REQUIRED'
        elif not all(conditions.get(k) is True for k in allowed):result['status']='ASSUMPTIONS_UNESTABLISHED'
        else:result['status']='REFERENCE_BACKED_CONDITIONAL_ARGUMENT'
    else:
        result['references']=[REFERENCES['kdf'],REFERENCES['hkdf']]
        result['claim']='Separated epoch transforms limit a naive solve that incorrectly treats multiple epochs as one unchanged key.'
        result['exclusions']=['This is an attack-specific accumulation argument, not a general confidentiality proof.',
                              'Rotation preserves norms/Gram information; shared subspaces and repeated plaintext remain attack surfaces.',
                              'Master-derived historical keys do not provide forward secrecy while the master is retained.',
                              'Independent epochs do not imply an irreversible key-erasure ratchet.']
        required={'separated_epoch_keys','enforced_epoch_budget','same_key_pooling_only'}
        result['status']='CONDITIONAL_NARROW_POOLING_ARGUMENT' if all(conditions.get(k) is True for k in required) else 'ASSUMPTIONS_UNESTABLISHED'
        if adversary in ('endpoint','pretermination','insider'):result['status']='STRONGER_ADVERSARY_REQUIRES_SEPARATE_ANALYSIS'
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('fresh output required')
    request=json.loads(args.input.read_text())
    if set(request)!={'mechanism','adversary','conditions'}:raise ValueError('exactly mechanism/adversary/conditions required')
    result=assess(**request);result['input_sha256']=file_sha256(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with EvidenceLogger(args.output):args.output.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
