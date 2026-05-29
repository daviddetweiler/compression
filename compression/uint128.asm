PUBLIC uint128_adj
PUBLIC get_subrange

_TEXT SEGMENT
; uint128_adj(range_width, ones_count, total)
uint128_adj:
	MOV RAX, RCX
	MUL RDX
	DIV R8
	RET

; write_bit(&state, bit)
write_bit:
	SHL QWORD PTR [RCX+48], 1
	OR [RCX+48], RDX
	ADD QWORD PTR [RCX+56], 1

	AND QWORD PTR [RCX+56], 63
	JNZ write_bit_done
	MOV RAX, [RCX+80]
	MOV R8, [RCX+56]
	MOV [RAX], R8
	ADD QWORD PTR [RCX+80], 8

write_bit_done:
	RET

; flush_bit(&state, bit)
flush_bit:
	CALL write_bit
	NOT RDX
	AND RDX, 1

flush_bit_again:
	MOV RAX, [RCX+64]
	TEST RAX, RAX
	JZ flush_bit_done
	CALL write_bit
	SUB QWORD PTR [RCX+64], 1
	JMP flush_bit_again

flush_bit_done:
	RET

; get_subrange(&state, bit)
; bit must be [0, 1]
; this is an extremely unreadable way to extract 
get_subrange:
	MOV R8, [RCX+32]
	SHL QWORD PTR [RCX+32], 1
	OR [RCX+32], RDX
	PEXT R8, R8, [RCX+16]
	MOV R9, [RCX+40]
	PEXT R9, R9, [RCX+24]
	MOV RAX, RCX
	POPCNT RCX, [RCX+16]
	SHL R9, CL
	MOV RCX, RAX
	OR R8, R9
	SHL R8, 4
	MOV R10, R8

	; At this point, r10 contains the current context model offset in bytes
	MOV RAX, [RCX+72]
	MOV R9, [RAX+R10+8]
	MOV R8, [RAX+R10]
	ADD [RAX+R10], RDX
	ADD QWORD PTR [RAX+R10+8], 1
	
	; Context model has been updated, r8 and r9 are the ones and total count, respectively
	MOV R11, RDX
	MOV RAX, [RCX+8]
	SUB RAX, [RCX]
	MOV R10, RAX
	MUL R8
	DIV R9
	MOV RDX, R11

	; RAX now contains the width of the ones range, while r10 has the original range width
	TEST RAX, RAX
	JNZ get_subrange_nzero
	ADD RAX, 1

get_subrange_nzero:
	CMP RAX, R10
	JNE get_subrange_ok
	SUB RAX, 1

get_subrange_ok:
	; Range has been clamped to ensure nonzero probability
	RET

_TEXT ENDS	

END