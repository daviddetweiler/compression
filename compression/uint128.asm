public uint128_adj
public get_subrange

_text segment
; uint128_adj(range_width, ones_count, total)
uint128_adj:
	mov rax, rcx
	mul rdx
	div r8
	ret

; write_bit(&state, bit)
write_bit:
	shl qword ptr [rcx + 56], 1
	or [rcx + 56], rdx
	add qword ptr [rcx + 64], 1

	and qword ptr [rcx + 64], 63
	jnz write_bit_done
	mov rax, [rcx + 80]
	mov r8, [rcx + 64]
	mov [rax], r8
	add qword ptr [rcx + 80], 8

write_bit_done:
	ret

; flush_bit(&state, bit)
flush_bit:
	call write_bit
	not rdx
	and rdx, 1

flush_bit_again:
	mov rax, [rcx + 72]
	test rax, rax
	jz flush_bit_done
	call write_bit
	sub qword ptr [rcx + 72], 1
	jmp flush_bit_again

flush_bit_done:
	ret

; get_model(&state, bit)
get_model:
	mov rax, [rcx + 32]
	shl qword ptr [rcx + 32], 1
	or [rcx + 32], rdx
	pext rax, rax, [rcx + 16]
	mov r9, [rcx + 40]
	pext r9, r9, [rcx + 24]
	mov r8, rcx
	popcnt rcx, [rcx + 16]
	shl r9, cl
	mov rcx, r8
	or rax, r9
	shl rax, 4
	ret

; get_subrange(&state, bit)
; bit must be [0, 1]
; this is an extremely unreadable way to extract 
get_subrange:
	call get_model
	mov r10, rax

	; at this point, r10 contains the current context model offset in bytes
	mov rax, [rcx + 48]
	mov r9, [rax + r10 + 8]
	mov r8, [rax + r10]
	add [rax + r10], rdx
	add qword ptr [rax + r10 + 8], 1
	
	; context model has been updated, r8 and r9 are the ones and total count, respectively
	mov r11, rdx
	mov rax, [rcx + 8]
	sub rax, [rcx]
	mov r10, rax
	mul r8
	div r9
	mov rdx, r11

	; rax now contains the width of the ones range, while r10 has the original range width
	test rax, rax
	jnz get_subrange_nzero
	add rax, 1

get_subrange_nzero:
	cmp rax, r10
	jne get_subrange_ok
	sub rax, 1

get_subrange_ok:
	; range has been clamped to ensure nonzero probability
	ret

_text ends	

end
