# Mecab 설치

Thanks to [최완재](https://github.com/mimi1942)

1. [Java JDK 설치](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
    - `java -version` 으로 설치 됐는지 확인
2. konlpy 설치 : `pip install konlpy`
    - `import konlpy` 했을 때 JPype 에러가 나면 다음처럼 라이브러리 설치
    - `pip install JPype1-py3`
3. Mecab 설치
    - sh파일 다운로드: `curl -o mecab_install.sh https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)`
    - 설치 파일 실행: `sh mecab_install.sh`
    - sh 파일은 남겨둘 필요 없으니 삭제: `rm mecab_install.sh`
4. 사용해보기

    ```py
    import konlpy
    from konlpy.tag import Mecab

    mecab = Mecab()
    test_seq = "여보세요 거기 누구없소, 거기 놓인 물 좀 내게 주오"
    parsed_seq = ["{}/{}".format(word, tag) for word, tag in mecab.pos(test_seq)]
    # ['여보세요/IC', '거기/NP', '누구/NP', '없/VA', '소/EF', ',/SC', '거기/NP', '놓인/VV+ETM', '물/NNG', '좀/MAG', '내/VV', '게/EC', '주/VX', '오/EC']
    ```
