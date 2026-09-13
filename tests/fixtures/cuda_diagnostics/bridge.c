// The bridge depends on the versioned SONAME, not the unversioned alias.
extern void fixture_increment(void);
void fixture_execute(void) { fixture_increment(); }
