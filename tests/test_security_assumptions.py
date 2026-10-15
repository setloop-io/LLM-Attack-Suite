import pytest
from attacker.security_assumptions import assess

TLS=dict(tls13=True,authenticated_peers=True,uncompromised_keys=True,trusted_implementation=True)

def test_tls_scopes_do_not_become_deployment_pass():
    passive=assess('tls13','passive_network',TLS)
    assert passive['status']=='REFERENCE_BACKED_CONDITIONAL_ARGUMENT'
    assert not passive['deployment_verified'] and not passive['counts_as_attack_coverage']
    for actor in ('endpoint','pretermination','insider'):
        assert assess('tls13',actor,TLS)['status']=='OUTSIDE_PROTECTED_ADVERSARY_SCOPE'
    assert assess('tls13','passive_network',dict(TLS,authenticated_peers=False))['status']=='ASSUMPTIONS_UNESTABLISHED'


def test_custom_sha_is_not_certified_by_standard_kdf_reference():
    assert assess('kdf','insider',dict(standard_hmac_kdf=False))['status']=='CUSTOM_CONSTRUCTION_REVIEW_REQUIRED'
    assert assess('kdf','passive_seam',dict(standard_hmac_kdf=True))['status']=='ASSUMPTIONS_UNESTABLISHED'


def test_ratchet_argument_is_narrow_and_preserves_master_compromise_limit():
    facts=dict(separated_epoch_keys=True,enforced_epoch_budget=True,same_key_pooling_only=True,master_retained=True)
    result=assess('ratchet','passive_seam',facts)
    assert result['status']=='CONDITIONAL_NARROW_POOLING_ARGUMENT'
    assert any('forward secrecy' in text for text in result['exclusions'])
    assert assess('ratchet','insider',facts)['status']=='STRONGER_ADVERSARY_REQUIRES_SEPARATE_ANALYSIS'
    assert assess('ratchet','passive_seam',dict(facts,same_key_pooling_only=False))['status']=='ASSUMPTIONS_UNESTABLISHED'


def test_unknown_or_string_conditions_fail_closed():
    with pytest.raises(ValueError):assess('tls13','passive_network',dict(tls13='true'))
    with pytest.raises(ValueError):assess('tls13','passive_network',dict(secure=True))
