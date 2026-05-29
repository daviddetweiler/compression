PUBLIC uint128_adj

_TEXT SEGMENT
; uint128_adj(range_width, ones_count, total)
uint128_adj:
	MOV RAX, RCX
	MUL RDX
	DIV R8
	RET

_TEXT ENDS	

END